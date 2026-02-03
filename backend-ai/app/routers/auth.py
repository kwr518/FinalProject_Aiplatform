from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
import requests
import os
import logging
from pydantic import BaseModel
import jwt

# .env 환경변수 로드
KAKAO_CLIENT_ID = os.getenv('KAKAO_CLIENT_ID')
KAKAO_CLIENT_SECRET = os.getenv('KAKAO_CLIENT_SECRET')

# 설정 주소
KAKAO_REDIRECT_URI = "http://localhost:8000/auth/kakao/callback" 
FRONTEND_URL = "http://localhost:3000" 
JAVA_USER_SYNC_URL = "http://localhost:8080/api/user/sync"

# 카카오 API URL
KAKAO_OAUTH_URL = 'https://kauth.kakao.com/oauth/authorize'
KAKAO_TOKEN_URL = 'https://kauth.kakao.com/oauth/token'
KAKAO_USER_INFO_URL = 'https://kapi.kakao.com/v2/user/me'
KAKAO_LOGOUT_URL = 'https://kapi.kakao.com/v1/user/logout'

router = APIRouter()
logger = logging.getLogger(__name__)

# ===== 헬퍼 함수 =====
def get_current_user(request: Request):
    """세션에서 사용자 정보 확인"""
    user = request.session.get('kakao_user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

def sync_user_with_java(user_info):
    """자바 서버로 유저 정보 동기화 (kakao_ 또는 google_ ID 포함)"""
    try:
        u_email = user_info.get('email') or ""
        
        # ID는 이미 접두사(kakao_, google_)가 붙은 상태로 넘어옴
        payload = {
            "loginSocialId": str(user_info.get('id')), 
            "userName": user_info.get('nickname'),
            "email": u_email,
            "safetyPortalId": "",
            "safetyPortalPw": ""
        }
        
        print(f"🚀 [Auth] 자바 서버로 전송: ID={payload['loginSocialId']}, Name={payload['userName']}")
        
        response = requests.post(JAVA_USER_SYNC_URL, json=payload, timeout=5)
        
        if response.status_code == 200:
            java_user = response.json()
            history_id = java_user.get('historyId')
            print(f"✅ [Auth] DB 저장/조회 성공! History ID: {history_id}")
            return history_id
        else:
            print(f"⚠️ [Auth] 자바 서버 응답 오류: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ [Auth] 자바 서버 연결 실패 (DB 저장 안됨): {e}")
        return None

# ===== 라우트 정의 =====

@router.get("/auth/kakao/login")
async def kakao_login():
    """1. 카카오 로그인 페이지로 리다이렉트"""
    if not KAKAO_CLIENT_ID:
        return JSONResponse({"error": "KAKAO_CLIENT_ID not set"}, status_code=500)
    
    params = {
        'client_id': KAKAO_CLIENT_ID,
        'redirect_uri': KAKAO_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'profile_nickname, account_email',
        'prompt': 'login' # 매번 로그인 창 뜨게 설정
    }
    login_url = f"{KAKAO_OAUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(login_url)

@router.get("/auth/kakao/callback")
async def kakao_callback(request: Request, code: str = None, error: str = None):
    """2. 카카오 인증 콜백 처리"""
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/?error={error}")
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/?error=no_code")

    try:
        # A. 토큰 발급
        token_res = requests.post(KAKAO_TOKEN_URL, data={
            'grant_type': 'authorization_code',
            'client_id': KAKAO_CLIENT_ID,
            'client_secret': KAKAO_CLIENT_SECRET,
            'code': code,
            'redirect_uri': KAKAO_REDIRECT_URI
        })
        token_json = token_res.json()
        
        if "access_token" not in token_json:
            return RedirectResponse(f"{FRONTEND_URL}/?error=token_failed")

        access_token = token_json['access_token']

        # B. 사용자 정보 요청
        user_res = requests.get(KAKAO_USER_INFO_URL, headers={
            "Authorization": f"Bearer {access_token}"
        })
        user_info = user_res.json()

        # C. 데이터 정리 (접두사 붙이기)
        kakao_account = user_info.get('kakao_account', {})
        profile = kakao_account.get('profile', {})
        
        social_id = f"kakao_{user_info.get('id')}" # ID 통일

        kakao_user = {
            'id': social_id,
            'nickname': profile.get('nickname', '사용자'),
            'email': kakao_account.get('email', ''),
            'profile_image': profile.get('thumbnail_image_url', ''),
            'access_token': access_token 
        }

        # D. 자바 서버 동기화
        hid = sync_user_with_java(kakao_user)
        if hid:
            kakao_user['history_id'] = hid 

        print(f"✅ [카카오 로그인] {kakao_user['nickname']} ({kakao_user['id']})")
        
        # E. 세션 저장 및 프론트로 이동
        request.session['kakao_user'] = kakao_user
        return RedirectResponse(url=FRONTEND_URL)

    except Exception as e:
        logger.error(f"Login failed: {e}")
        return RedirectResponse(f"{FRONTEND_URL}/?error=server_error")

@router.get("/api/auth/check")
async def check_auth(request: Request):
    """3. 프론트엔드에서 로그인 상태 확인용"""
    user = request.session.get('kakao_user')
    if user:
        return {"authenticated": True, "user": user}
    return {"authenticated": False, "user": None}

@router.post("/auth/logout")
async def logout(request: Request):
    """4. 로그아웃"""
    user = request.session.get('kakao_user')

    if user and 'access_token' in user:
        # 카카오 토큰일 경우만 카카오 서버 로그아웃 시도
        if str(user['id']).startswith('kakao_'):
            try:
                requests.post(KAKAO_LOGOUT_URL, headers={
                    "Authorization": f"Bearer {user['access_token']}"
                })
            except:
                pass
            
    request.session.clear()
    return {"success": True}

# 구글 로그인 요청 바디 정의
class GoogleLoginRequest(BaseModel):
    token: str

@router.post("/api/auth/google")
async def google_login_endpoint(request: Request, body: GoogleLoginRequest):
    """5. 구글 로그인 처리"""
    try:
        token = body.token
        # 서명 검증 없이 디코딩 (프론트에서 이미 검증되었다고 가정)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # A. 데이터 정리 (접두사 붙이기)
        user_info = {
            'id': f"google_{decoded.get('sub')}", 
            'nickname': decoded.get('name', 'Google User'),
            'email': decoded.get('email', ''),
            'profile_image': decoded.get('picture', ''),
            'access_token': 'google_token_dummy' # 구글은 액세스 토큰 방식이 다르므로 더미값
        }

        # B. 자바 서버 동기화
        hid = sync_user_with_java(user_info)
        if hid:
            user_info['history_id'] = hid

        # C. 세션 저장 (키 이름은 편의상 kakao_user로 통일 유지)
        request.session['kakao_user'] = user_info 
        
        print(f"✅ [구글 로그인] {user_info['nickname']} ({user_info['id']})")
        return {"result": "success", "user": user_info}

    except Exception as e:
        print(f"❌ 구글 로그인 실패: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)