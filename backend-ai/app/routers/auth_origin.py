from fastapi import APIRouter, Request, Response, HTTPException, Depends
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.config import Config
from starlette.requests import Request
import requests
import os
from datetime import datetime, timedelta
import logging

# .env 환경변수 로드 (main.py에서 이미 load_dotenv() 했다고 가정)
KAKAO_CLIENT_ID = os.getenv('KAKAO_CLIENT_ID')
KAKAO_CLIENT_SECRET = os.getenv('KAKAO_CLIENT_SECRET')
# ⚠️ 중요: 카카오 개발자 센터 Redirect URI를 이 주소로 설정해야 합니다. (포트 8000)
KAKAO_REDIRECT_URI = "http://localhost:8000/auth/kakao/callback" 
FRONTEND_URL = "http://localhost:8080" # 로그인 완료 후 돌아갈 스프링부트 주소

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
        'scope': 'profile_nickname' # 필요한 권한 추가
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

        # FastAPI 세션에 저장
        request.session['kakao_user'] = kakao_user
        
        # 로그인 성공 후 프론트엔드(스프링부트) 홈으로 이동
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