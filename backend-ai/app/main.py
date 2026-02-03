import os
import shutil
import requests
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel

# 기존 라우터 임포트
from app.routers import traffic, auth 

# 서비스 모듈 안전하게 임포트
try:
    from app.services.s3_service import s3_manager
    from app.services.ai_service import ai_manager
except ImportError:
    s3_manager = None
    ai_manager = None
    print("❌ [오류] 서비스 모듈(s3_service, ai_service)을 찾을 수 없습니다.")

app = FastAPI(title="AI 교통관제 시스템")

# 1. 세션 미들웨어 (카카오 로그인용)
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# 2. CORS 설정 (프론트엔드 및 자바 서버 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080", 
        "http://127.0.0.1:8080",
        "http://localhost:3000",   
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 라우터 등록
app.include_router(traffic.router) 
app.include_router(auth.router)    

# 임시 파일 저장소
TEMP_DIR = "temp_videos"
os.makedirs(TEMP_DIR, exist_ok=True)

# 자바 서버 주소
JAVA_SERVER_URL = "http://localhost:8080/api/violations"

@app.get("/")
def read_root():
    ocr_status = "✅ 로드됨" if (ai_manager and ai_manager.lpr_system) else "❌ 로드 안됨"
    return {
        "status": "running", 
        "message": "AI 관제 시스템 가동 중", 
        "ocr_module": ocr_status
    }

# ★ 백그라운드 작업 함수 (통합됨)
def background_s3_upload(local_path: str, s3_key: str):
    """파일을 S3에 업로드하고 로컬 파일을 삭제하는 백그라운드 작업"""
    if s3_manager:
        try:
            print(f"☁️ [Background] S3 업로드 시작: {s3_key}")
            s3_manager.upload_file(local_path, s3_key)
            print(f"✅ [Background] S3 업로드 완료")
        except Exception as e:
            print(f"❌ [Background] S3 업로드 실패: {e}")
    
    # 업로드 후 로컬 파일 삭제 (서버 용량 관리)
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
            print(f"🗑️ [Background] 임시 파일 삭제 완료")
        except:
            pass

# ★ 분석 엔드포인트 (통합 및 정리됨)
@app.post("/api/analyze-video")
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    serial_no: str = Form(...) # 프론트에서 보낸 serial_no 받기
):
    if ai_manager is None:
        return JSONResponse(content={"result": "AI 모듈 로드 실패", "plate": "Error"}, status_code=500)

    # 1. 파일 저장
    filename = file.filename
    file_path = os.path.join(TEMP_DIR, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 폴더명 결정 (없으면 WEB_UPLOAD)
        folder_name = serial_no if serial_no else "WEB_UPLOAD"
        print(f"📥 [Main] 영상 수신: {filename} (저장 폴더: {folder_name})")

        # 2. AI 분석 실행
        print("🔄 AI 분석 엔진 가동 (YOLO + TF)...")
        result = ai_manager.analyze_local_video(file_path)
        
        # 3. S3 경로(Key) 생성
        # raspberrypi_video 폴더 안에 -> 시리얼번호 폴더 안에 -> 파일
        s3_key = f"raspberrypi_video/{folder_name}/{filename}"
        
        if s3_manager:
            # 미리보기 URL 생성 (업로드 전이라도 미리 생성 가능)
            result["video_url"] = s3_manager.get_presigned_url(s3_key)
        
        print(f"✅ [Main] 분석 완료: {result['result']}")

        # 4. 자바 서버로 결과 전송 (DB 저장용)
        try:
            # 자바 쪽에도 시리얼 번호 같이 넘겨줌
            result["serial_no"] = folder_name
            
            print(f"🚀 [Main] 자바 서버로 데이터 전송 시도: {JAVA_SERVER_URL}")
            response = requests.post(JAVA_SERVER_URL, json=result, timeout=5)
            
            if response.status_code == 200:
                print("✅ [Main] 자바 서버 DB 저장 성공!")
            else:
                print(f"⚠️ [Main] 자바 서버 응답 오류: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ [Main] 자바 서버 연결 실패 (DB 저장 안됨): {e}")

        # 5. S3 업로드는 백그라운드로 넘김 (응답 속도 향상)
        background_tasks.add_task(background_s3_upload, file_path, s3_key)

        # 6. 프론트엔드에 결과 반환
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ [Main] 서버 에러: {str(e)}")
        # 에러 나면 파일 지우기
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return JSONResponse(content={
            "result": "서버 오류",
            "plate": "Error",
            "description": str(e)
        }, status_code=500)

# 영상 삭제 요청 모델
class DeleteVideoRequest(BaseModel):
    video_url: str

@app.post("/api/delete-video")
def delete_video_endpoint(req: DeleteVideoRequest):
    if not s3_manager:
        return JSONResponse({"error": "S3 Manager not loaded"}, status_code=500)
    
    try:
        # URL에서 S3 Key 추출 로직
        url = req.video_url
        if "raspberrypi_video" in url:
            # URL 디코딩 및 파싱 로직 (단순화)
            start_idx = url.find("raspberrypi_video")
            end_idx = url.find("?")
            
            if end_idx == -1:
                key = url[start_idx:]
            else:
                key = url[start_idx:end_idx]
            
            print(f"🗑️ [S3 삭제 요청] Key: {key}")
            # s3_service.py에 delete_file 메서드 호출
            s3_manager.delete_file(key) 
            return {"status": "deleted", "key": key}
        else:
            print("⚠️ S3 키를 찾을 수 없는 URL입니다.")
            return {"status": "skipped"}
            
    except Exception as e:
        print(f"❌ S3 삭제 중 에러: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)