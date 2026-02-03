from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.config import Config
from starlette.requests import Request
import requests
import os
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel
import jwt

# .env 환경변수 로드
KAKAO_CLIENT_ID = os.getenv('KAKAO_CLIENT_ID')
KAKAO_CLIENT_SECRET = os.getenv('KAKAO_CLIENT_SECRET')
KAKAO_REDIRECT_URI = "http://localhost:8000/auth/kakao/callback" 
FRONTEND_URL = "http://localhost:3000" 

# 자바 서버 유저 동기화 주소
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
    user = request.session.get('kakao_user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

# ★ [수정됨] 자바 서버 동기화 (전화번호 로직 완전 제거)
def sync_user_with_java(user_info):
    try:
        # 이메일 안전하게 가져오기
        u_email = user_info.get('email') or ""
        
        # UserDTO와 필드명 일치 (userNumber 삭제됨)
        payload = {
            "loginSocialId": str(user_info.get('id')), # 여기서 이미 kakao_ 붙은 상태로 옴
            "userName": user_info.get('nickname'),
            "email": u_email,
            "safetyPortalId": "",
            "safetyPortalPw": ""
        }
        
        # 로그에서 userNumber 제거 (이게 에러 원인이었음)
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
    if not KAKAO_CLIENT_ID:
        return JSONResponse({"error": "KAKAO_CLIENT_ID not set"}, status_code=500)
    
    params = {
        'client_id': KAKAO_CLIENT_ID,
        'redirect_uri': KAKAO_REDIRECT_URI,
        'response_type': 'code',
        'scope': 'profile_nickname, account_email', 
        'prompt': 'login' 
    }
    login_url = f"{KAKAO_OAUTH_URL}?" + "&".join(f"{k}={v}" for k, v in params.items())
    return RedirectResponse(login_url)

@router.get("/auth/kakao/callback")
async def kakao_callback(request: Request, code: str = None, error: str = None):
    if error:
        return RedirectResponse(f"{FRONTEND_URL}/?error={error}")
    if not code:
        return RedirectResponse(f"{FRONTEND_URL}/?error=no_code")

    try:
        # 토큰 발급
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

        kakao_account = user_info.get('kakao_account', {})
        profile = kakao_account.get('profile', {})

        # ★ [핵심] ID에 접두사 'kakao_' 붙이기
        social_id = f"kakao_{user_info.get('id')}"

        kakao_user = {
            'id': social_id, 
            'nickname': profile.get('nickname', '사용자'),
            'email': kakao_account.get('email', ''),
            'profile_image': profile.get('thumbnail_image_url', ''),
            'access_token': access_token 
        }

        # 자바 DB 동기화
        hid = sync_user_with_java(kakao_user)
        if hid:
            kakao_user['history_id'] = hid 

        print(f"✅ [로그인 성공] {kakao_user['nickname']} ({kakao_user['id']})")
        
        request.session['kakao_user'] = kakao_user
        return RedirectResponse(url=FRONTEND_URL)

    except Exception as e:
        logger.error(f"Login failed: {e}")
        return RedirectResponse(f"{FRONTEND_URL}/?error=server_error")

@router.get("/api/auth/check")
async def check_auth(request: Request):
    user = request.session.get('kakao_user')
    if user:
        return {"authenticated": True, "user": user}
    return {"authenticated": False, "user": None}

@router.post("/auth/logout")
async def logout(request: Request):
    user = request.session.get('kakao_user')
    if user and 'access_token' in user:
        try:
            requests.post(KAKAO_LOGOUT_URL, headers={
                "Authorization": f"Bearer {user['access_token']}"
            })
        except:
            pass
    request.session.clear()
    return {"success": True}

class GoogleLoginRequest(BaseModel):
    token: str

@router.post("/api/auth/google")
async def google_login_endpoint(request: Request, body: GoogleLoginRequest):
    try:
        token = body.token
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # ★ [핵심] 구글은 이미 google_ 접두사를 붙여서 처리 중
        user_info = {
            'id': f"google_{decoded.get('sub')}", 
            'nickname': decoded.get('name', 'Google User'),
            'email': decoded.get('email', ''),
            'profile_image': decoded.get('picture', ''),
            'access_token': 'google_token_dummy'
        }

        hid = sync_user_with_java(user_info)
        if hid:
            user_info['history_id'] = hid

        request.session['kakao_user'] = user_info 
        return {"result": "success", "user": user_info}

    except Exception as e:
        print(f"❌ 구글 로그인 실패: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)