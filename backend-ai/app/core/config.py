import os
from botocore.config import Config
from dotenv import load_dotenv  # 👈 추가됨

# .env 파일의 내용을 환경 변수로 로드합니다.
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- [파일 경로 설정] ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "test_model.h5")
YOLO_PATH = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
CSV_FILE = "violations_log.csv"
TEMP_VIDEO_DIR = "temp_videos"

if not os.path.exists(TEMP_VIDEO_DIR):
    os.makedirs(TEMP_VIDEO_DIR)

# --- [AI 파라미터] ---
SEQUENCE_LENGTH = 50
STEP_SIZE = 10
CATEGORIES = ['신호위반', '중앙선침범', '진로변경위반']

# --- [AWS S3 설정] ---
# .env에 적힌 변수명과 일치시켜야 합니다.
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "human-final-project-bucket")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")

# S3 Config 설정
S3_CONFIG = Config(region_name=AWS_REGION, signature_version='s3v4')

# --- [자바 서버 연동 설정] ---
USE_JAVA_SYNC = True
# 💡 로컬 테스트를 위해 192.168.0.40 주소로 변경합니다.
JAVA_SERVER_URL = "http://192.168.0.40:8080/api/violations"