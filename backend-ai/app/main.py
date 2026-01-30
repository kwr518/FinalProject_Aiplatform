import os
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware 
from fastapi.middleware.cors import CORSMiddleware 

# 기존 라우터 임포트 (로그인, 대시보드)
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

# 2. CORS 설정 (프론트엔드 연동용)
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

# 헬스 체크 엔드포인트
@app.get("/")
def read_root():
    ocr_status = "✅ 로드됨" if (ai_manager and ai_manager.lpr_system) else "❌ 로드 안됨"
    return {
        "status": "running", 
        "message": "AI 관제 시스템 가동 중", 
        "ocr_module": ocr_status
    }

# ★ 백그라운드 작업 함수 (리액트에 응답을 보낸 뒤에 실행됨)
def background_s3_upload(local_path: str, filename: str):
    if s3_manager:
        try:
            # 사용자가 원하던 'raspberrypi_video/' 폴더로 고정
            s3_key = f"raspberrypi_video/{filename}"
            print(f"☁️ [Background] S3 업로드 시작: {s3_key}")
            
            s3_manager.upload_file(local_path, s3_key)
            print(f"✅ [Background] S3 업로드 완료")
            
        except Exception as e:
            print(f"❌ [Background] S3 업로드 실패: {e}")
    
    # 업로드가 끝나면 로컬 파일 삭제 (서버 용량 관리)
    if os.path.exists(local_path):
        try:
            os.remove(local_path)
            print(f"🗑️ [Background] 임시 파일 삭제 완료: {filename}")
        except:
            pass

# ★ 분석 엔드포인트 (파일명 수정됨 & 백그라운드 적용됨)
@app.post("/api/analyze-video")
async def analyze_video_endpoint(
    background_tasks: BackgroundTasks, # 👈 백그라운드 태스크 기능 추가
    file: UploadFile = File(...)
):
    if ai_manager is None:
        return JSONResponse(content={"result": "AI 모듈 로드 실패", "plate": "Error"}, status_code=500)

    try:
        # 1. 파일명 그대로 사용 (접두사 'upload_' 제거!)
        filename = file.filename
        file_path = os.path.join(TEMP_DIR, filename)
        
        # 2. 로컬에 일단 저장
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"📥 [Main] 영상 수신: {filename}")

        # 3. AI 분석 실행 (사용자가 기다리는 핵심 작업)
        print("🔄 AI 분석 엔진 가동 (YOLO + TF)...")
        result = ai_manager.analyze_local_video(file_path)
        
        # 4. S3 URL 생성 (업로드는 뒤에서 하더라도 주소 규칙은 아니까 미리 생성)
        if s3_manager:
            s3_key = f"raspberrypi_video/{filename}"
            # 파일이 아직 안 올라갔어도 URL은 미리 만들 수 있음
            result["video_url"] = s3_manager.get_presigned_url(s3_key)
        
        print(f"✅ [Main] 분석 완료, 결과 반환: {result}")

        # 5. [중요] S3 업로드는 '나중에 해'라고 등록 (리액트 스피너 멈추게 하기 위함)
        # 로컬 파일 경로와 파일명을 넘겨줌
        background_tasks.add_task(background_s3_upload, file_path, filename)

        # 6. 결과 즉시 반환 (여기서 리액트 스피너가 멈춤!)
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