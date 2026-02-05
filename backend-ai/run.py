import uvicorn
import os
import sys
import boto3
from dotenv import load_dotenv

# 같은 폴더내에 있는 .env 파일 로드
load_dotenv()

# 환경변수 읽기
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
NGROK_TOKEN = os.getenv("NGROK_AUTHTOKEN")

# ⚠️ 본인의 AWS Lambda 함수 이름을 정확히 적어주세요!
LAMBDA_FUNCTION_NAME = "s3-to-fastapi" 

try:
    from pyngrok import ngrok
except ImportError:
    print("❌ pyngrok가 설치되지 않았습니다. 'pip install pyngrok'를 실행하세요.")
    sys.exit(1)

def update_lambda_env(public_url):
    """AWS Lambda의 환경변수를 자동으로 수정하는 함수"""
    print("⏳ AWS Lambda 설정 업데이트 중...")
    try:
        client = boto3.client('lambda', # python에서 AWS 서비스를 조작할 떄 쓰는 라이브러리 Boto3를 이용해 조종키 생성
            # .env 파일에서 읽어온 키들을 이용해 계정 인증 및 람다가 위치한 지역을 지정
            aws_access_key_id=AWS_ACCESS_KEY,       
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        
        # Lambda 환경변수 업데이트(이미 만들어진 람다 함수의 설정을 실시간으로 변경)
        client.update_function_configuration(
            FunctionName=LAMBDA_FUNCTION_NAME,  # 수많은 람다 함수중 어떤 것을 고칠지 이름 지정
            Environment={   # 람다 함수 내부의 환경변수 탭에 있는 값을 수정
                'Variables': {
                    'TARGET_URL': public_url    # 람다가 분석 요청을 보낼 목적지 주소인 TARGET_URL이라는 변수에 ngrok이 만든 새 주소를 덮어쓰기
                }
            }
        )
        
        # 서버 실행 때마다 변수 이름은 그대로지만 ngrok 주소는 서버를 킬때마다 변경되므로  위 코드를 이용해 매번 자동 업데이트 진행
        
        print(f"✅ AWS Lambda [{LAMBDA_FUNCTION_NAME}] 주소 업데이트 완료!")
        print(f"   새 타겟: {public_url}")
        
    except Exception as e:
        print(f"❌ AWS 설정 실패: {e}")

# 파일 실행시 하위 코드 실행
if __name__ == "__main__":
    PORT = 8000     # 포트 8000으로 고정
    
    # 1. Ngrok 인증 및 터널 열기
    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)   # .env에 저장된 Ngrok 토큰을 등록
    
    try:
        ngrok.kill()    # 이전에 켜두었던 ngrok이 남아있을 경우가 있어 깨끗히 종료 하고 실행함
        # 8000번 포트로 통하는 외부 인터넷 전용 가상 주소 생성
        # http://random.ngrok-free.app 같은 주소가 생성되어 public_url 변수에 담김
        public_url = ngrok.connect(PORT).public_url     
        
        print("=" * 60)
        print(f"🚀 Ngrok 터널 개방: {public_url}")
        
        # 새로운 주소를 AWS Lambda에게 전달
        update_lambda_env(public_url)
        
        print("=" * 60)

        # FastAPI 서버를 실행해 
        # 내 컴퓨터에서 실행한 이후 ngrok이 외부 신호를 
        uvicorn.run("app.main:app",     # app 폴더 안의 main.py 파일에 있는 app 객체 실행
                    host="0.0.0.0",   # 127.0.0.1:8000 내부 주소와 외부 주소를 연결
                    port=PORT, 
                    reload=False)       # run.py 에서 실행 때 코드 수정을 실시간으로 반영하지 않음(배포/실행 환경)
    except Exception as e:
        print(f"❌ 실행 에러: {e}")