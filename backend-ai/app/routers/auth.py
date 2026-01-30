from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.config import Config
from starlette.requests import Request
import requests
import os
from datetime import datetime, timedelta
import logging

# .env 환경변수 로드
KAKAO_CLIENT_ID = os.getenv('KAKAO_CLIENT_ID')
KAKAO_CLIENT_SECRET = os.getenv('KAKAO_CLIENT_SECRET')
# ⚠️ 중요: 카카오 개발자 센터 Redirect URI를 이 주소로 설정해야 합니다. (포트 8000)
KAKAO_REDIRECT_URI = "http://localhost:8000/auth/kakao/callback" 
# ⚠️ 중요: 프론트엔드 주소 (React는 3000번입니다. 8080은 자바 서버이므로 3000으로 수정 추천)
FRONTEND_URL = "http://localhost:3000" 

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
        'scope': 'profile_nickname',
        # ★ [추가됨] 이 옵션이 있으면 매번 아이디/비번 입력창이 강제로 뜹니다 ★
        'prompt': 'login' 
    }
    # URL 생성
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
        # 토큰 발급 요청
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

        # 사용자 정보 요청
        user_res = requests.get(KAKAO_USER_INFO_URL, headers={
            "Authorization": f"Bearer {access_token}"
        })
        user_info = user_res.json()

        # 세션에 저장할 데이터 정리
        kakao_user = {
            'id': user_info.get('id'),
            'nickname': user_info.get('kakao_account', {}).get('profile', {}).get('nickname', '사용자'),
            'email': user_info.get('kakao_account', {}).get('email', ''),
            'profile_image': user_info.get('kakao_account', {}).get('profile', {}).get('thumbnail_image_url', ''),
            'access_token': access_token # 로그아웃을 위해 저장
        }

        # ★ [요청하신 로그 출력 코드 추가됨] ★
        print("\n" + "="*60)
        print(f" [로그인 성공] 카카오에서 받은 유저 정보:")
        print(f" ID: {kakao_user['id']}")
        print(f" 닉네임: {kakao_user['nickname']}")
        print(f" 이메일: {kakao_user['email']}")
        print(f" 프사: {kakao_user['profile_image']}")
        print("="*60 + "\n")

        # FastAPI 세션에 저장
        request.session['kakao_user'] = kakao_user
        
        # 로그인 성공 후 프론트엔드로 이동
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
        # 카카오 서버에서도 로그아웃 (선택사항)
        try:
            requests.post(KAKAO_LOGOUT_URL, headers={
                "Authorization": f"Bearer {user['access_token']}"
            })
        except:
            pass
            
    request.session.clear()
    return {"success": True}


from pydantic import BaseModel
import jwt # pip install pyjwt 필요 (보통 설치되어 있음)

class GoogleLoginRequest(BaseModel):
    token: str

@router.post("/api/auth/google")
async def google_login_endpoint(request: Request, body: GoogleLoginRequest):
    try:
        # 1. 구글 토큰 디코딩 (검증은 생략하고 내용만 추출 - 실제 서비스에선 verify 필수)
        # 구글 토큰은 JWT 형식이므로, 내용을 까보면 이메일/이름이 들어있습니다.
        token = body.token
        
        # jwt.decode는 verify=False로 하면 서명 검증 없이 내용만 봅니다 (개발용)
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # 2. 세션 데이터 구성 (기존 카카오 구조와 통일)
        user_info = {
            'id': f"google_{decoded.get('sub')}", # 구글 고유 ID
            'nickname': decoded.get('name', 'Google User'),
            'email': decoded.get('email', ''),
            'profile_image': decoded.get('picture', ''),
            'access_token': 'google_token_dummy' # 구글은 액세스 토큰 방식이 다르지만 형식 유지
        }

        # 3. 세션 저장
        request.session['kakao_user'] = user_info # 키 이름을 kakao_user로 통일해야 기존 로직과 호환됨
        
        print(f"✅ 구글 로그인 성공: {user_info['nickname']}")
        return {"result": "success", "user": user_info}

    except Exception as e:
        print(f"❌ 구글 로그인 처리 실패: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)