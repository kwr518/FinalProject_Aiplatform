import os
import shutil
import requests 
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Form # ★ Form 임포트 필수
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware 
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel # ★ 추가

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

# 1. 세션 미들웨어
app.add_middleware(SessionMiddleware, secret_key="your-secret-key-here")

# 2. CORS 설정
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

# ★ [수정 1] 백그라운드 함수가 'filename'이 아니라 's3_key'(전체 경로)를 받도록 변경
def background_s3_upload(local_path: str, s3_key: str):
    if s3_manager:
        try:
            # 여기서 경로를 또 만들면 안 됨. 인자로 받은 s3_key를 그대로 사용.
            print(f"☁️ [Background] S3 업로드 시작: {s3_key}")
            s3_manager.upload_file(local_path, s3_key)
            print(f"✅ [Background] S3 업로드 완료")
        except Exception as e:
            print(f"❌ [Background] S3 업로드 실패: {e}")
    
    # 업로드 후 로컬 파일 삭제
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
        except:
            pass

# ★ [수정 2] 분석 엔드포인트: 시리얼 번호로 경로 생성
@app.post("/api/analyze-video")
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    serial_no: str = Form(...) # 프론트에서 보낸 serial_no 받기
):
    if ai_manager is None:
        return JSONResponse(content={"result": "AI 모듈 로드 실패", "plate": "Error"}, status_code=500)

    try:
        # 1. 파일 저장
        filename = file.filename
        file_path = os.path.join(TEMP_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 폴더명 결정 (없으면 WEB_UPLOAD)
        folder_name = serial_no if serial_no else "WEB_UPLOAD"
        print(f"📥 [Main] 영상 수신: {filename} (저장 폴더: {folder_name})")

        # 2. AI 분석 실행
        print("🔄 AI 분석 엔진 가동 (YOLO + TF)...")
        result = ai_manager.analyze_local_video(file_path)
        
        # 3. S3 경로(Key) 생성 ★ 여기가 핵심입니다 ★
        # raspberrypi_video 폴더 안에 -> 시리얼번호 폴더 안에 -> 파일
        s3_key = f"raspberrypi_video/{folder_name}/{filename}"
        
        if s3_manager:
            # 미리보기 URL 생성
            result["video_url"] = s3_manager.get_presigned_url(s3_key)
        
        print(f"✅ [Main] 분석 완료: {result['result']}")

        # 4. 자바 서버로 결과 전송
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

        # 5. S3 업로드는 백그라운드로 넘김 (완성된 s3_key 전달)
        background_tasks.add_task(background_s3_upload, file_path, s3_key)

        # 6. 프론트엔드에 결과 반환
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ [Main] 서버 에러: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)
            
        return JSONResponse(content={
            "result": "서버 오류",
            "plate": "Error",
            "description": str(e)
        }, status_code=500)
        

class DeleteVideoRequest(BaseModel):
    video_url: str

@app.post("/api/delete-video")
def delete_video_endpoint(req: DeleteVideoRequest):
    if not s3_manager:
        return JSONResponse({"error": "S3 Manager not loaded"}, status_code=500)
    
    try:
        # URL에서 S3 Key 추출 (presigned url 등에서 key 부분만 발췌)
        # 예: https://bucket.../raspberrypi_video/WEB_UPLOAD/file.mp4?...
        # 간단하게 'raspberrypi_video' 뒷부분을 찾습니다.
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
            s3_manager.delete_file(key) # s3_service.py에 delete_file 메서드가 있어야 함 (보통 boto3 delete_object)
            return {"status": "deleted", "key": key}
        else:
            print("⚠️ S3 키를 찾을 수 없는 URL입니다.")
            return {"status": "skipped"}
            
    except Exception as e:
        print(f"❌ S3 삭제 중 에러: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)