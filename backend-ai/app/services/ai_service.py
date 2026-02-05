import os
import cv2
import numpy as np
import tensorflow as tf
import requests
import urllib.parse
from datetime import datetime
from ultralytics import YOLO 
from app.core.config import (
    MODEL_PATH, YOLO_PATH, SEQUENCE_LENGTH, STEP_SIZE, 
    CATEGORIES, CSV_FILE, TEMP_VIDEO_DIR,
    USE_JAVA_SYNC, JAVA_SERVER_URL
)
from app.core.global_state import detection_logs
from app.services.s3_service import s3_manager
from app.services.llm_service import get_llm_manager  # ★ 1. LLM 매니저 가져오기

# 번호판 인식 모듈 (선택적 로드)
try:
    from .plate_ocr import PlateRecognizerModule
except ImportError:
    PlateRecognizerModule = None

# 학습시킨 모델 경로 설정
base_dir = os.path.dirname(os.path.dirname(__file__))
NEW_YOLO_PATH = os.path.join(base_dir, "models", "best.pt") 

processing_files = set()

class AIService:
    def __init__(self):
        # 1. 위반 감지 모델 (TensorFlow - .h5)
        print("⏳ TF 모델 로딩 중...")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            print("✅ TF 모델 로드 완료")
        except Exception as e:
            print(f"❌ TF 모델 로드 실패: {e}")
            self.model = None
        
        # 2. 학습된 YOLO 모델 로드 (.pt)
        print(f"⏳ YOLO 학습 모델 로딩 중: {NEW_YOLO_PATH}")
        try:
            self.obj_detector = YOLO(NEW_YOLO_PATH)
            print("✅ YOLO 객체 탐지 모델 로드 완료")
        except Exception as e:
            print(f"❌ YOLO 로드 실패: {e}")
            self.obj_detector = None

        # 3. 번호판 인식기
        try:
            if PlateRecognizerModule:
                self.lpr_system = PlateRecognizerModule(YOLO_PATH) 
                print("✅ 번호판 인식 시스템 로드 완료")
            else:
                self.lpr_system = None
                print("⚠️ 번호판 모듈 없음 (Import 실패)")
        except Exception as e:
            print(f"❌ 번호판 모듈 초기화 실패: {e}")
            self.lpr_system = None

    def analyze_local_video(self, local_path):
        """자바 서버에서 전달받은 로컬 파일을 직접 분석하는 메서드"""
        try:
            filename = os.path.basename(local_path)
            cap = cv2.VideoCapture(local_path)
            all_frames = []
            detected_items = set() 

            print(f"🔄 AI 분석 엔진 가동 (YOLO + TF): {filename}")

            while True:
                ret, frame = cap.read()
                if not ret: break

                # 1. YOLO(.pt) 실시간 탐지
                if self.obj_detector:
                    # conf=0.4: 확신도 40% 이상만 감지
                    results = self.obj_detector(frame, conf=0.4, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            # 클래스 ID를 이름으로 변환
                            name = self.obj_detector.names[int(box.cls[0])]
                            detected_items.add(name)

                # 프레임 전처리 (TF 모델용)
                # 모델 입력 크기(128x128)에 맞춰 리사이즈 및 정규화
                all_frames.append(cv2.resize(frame, (128, 128)) / 255.0)
            
            cap.release()

            # 2. 위반 판단 (TensorFlow - .h5 모델)
            if len(all_frames) < SEQUENCE_LENGTH:
                return {"result": "분석 불가(영상 짧음)", "prob": 0, "plate": "-"}

            # 시퀀스 생성
            windows = [all_frames[i : i + SEQUENCE_LENGTH] for i in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, STEP_SIZE)]
            
            # 예측 수행
            if not windows:
                 return {"result": "분석 불가(프레임 부족)", "prob": 0, "plate": "-"}
                 
            predictions = self.model.predict(np.array(windows), batch_size=2, verbose=0)
            
            # 최고 확률 구간 찾기
            best_prob, best_class_idx, best_window_idx = 0, -1, -1
            for i, pred in enumerate(predictions):
                idx = np.argmax(pred)
                if pred[idx] > best_prob:
                    best_prob, best_class_idx, best_window_idx = pred[idx], idx, i

            # =========================================================
            # 🚀 정상 주행 필터링 (임계값 적용)
            # =========================================================
            MIN_CONFIDENCE = 0.5  # 50% 미만이면 위반 아님(정상)으로 간주

            if best_prob < MIN_CONFIDENCE:
                raw_label = "정상 주행"
                best_window_idx = -1 # 정상 주행이므로 번호판 인식 스킵 유도
            else:
                raw_label = CATEGORIES[best_class_idx] if best_class_idx != -1 else "정상 주행"

            # 3. 결과 정리
            obj_summary = ", ".join(list(detected_items)) if detected_items else "없음"
            final_display_result = f"{raw_label}" # 위반명만 사용

            # 4. 번호판 인식 (위반이 감지된 경우에만 수행)
            plate_text = "-"
            if self.lpr_system and best_window_idx != -1:
                # 위반 발생 구간의 프레임 인덱스 계산
                start_frame = best_window_idx * STEP_SIZE
                # 해당 구간 OCR 수행
                plate_text = self.lpr_system.process_segment(local_path, start_frame, SEQUENCE_LENGTH) or "인식 불가"

            return {
                "result": final_display_result, 
                "plate": plate_text,
                "location": "--", # GPS 연동 전 임시값
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "prob": round(float(best_prob * 100), 2),
                "info": f"YOLO 감지: {obj_summary}",
                "video_url": "" 
            }

        except Exception as e:
            print(f"❌ 로컬 분석 에러: {e}")
            # import traceback
            # traceback.print_exc()
            return {"result": "에러 발생", "prob": 0, "plate": "Error"}

    def process_video_task(self, video_key):
        """S3 업로드 시 백그라운드 분석 태스크"""
        # URL 디코딩 (한글 파일명 처리)
        decoded_key = urllib.parse.unquote_plus(video_key)
        filename = os.path.basename(decoded_key)

        if filename in processing_files: return
        processing_files.add(filename)

        try:
            local_path = os.path.join(TEMP_VIDEO_DIR, filename)
            
            # 폴더가 없으면 생성
            os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
            
            s3_manager.download_file(decoded_key, local_path)
            
            # 1. 영상 분석 수행
            analysis_result = self.analyze_local_video(local_path)
            video_url = s3_manager.get_presigned_url(decoded_key)
            
            # 날짜 및 시간 분리 (Java DTO 포맷 맞춤)
            incident_datetime = analysis_result.get("time", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            try:
                dt_obj = datetime.strptime(incident_datetime, '%Y-%m-%d %H:%M:%S')
                incident_date = dt_obj.strftime('%Y-%m-%d')
                incident_time = dt_obj.strftime('%H:%M:%S')
            except:
                incident_date = incident_datetime
                incident_time = ""

            # 시리얼 번호 (파일명 활용)
            serial_no = os.path.splitext(filename)[0]
            violation_type = analysis_result.get("result", "")

            # ---------------- [추가된 코드 시작: LLM 신고 초안 생성] ----------------
            # 2. LLM 매니저 가져오기
            llm_manager = get_llm_manager()
            ai_description = ""
            
            # 위반 사항이 있을 때만 초안 생성 ('정상 주행'이나 '에러'가 아닐 때)
            if "정상" not in violation_type and "에러" not in violation_type:
                # AI에게 던져줄 프롬프트 만들기
                draft_prompt = f"""
                다음 위반 사실을 바탕으로 안전신문고 신고 내용을 "상세 내용" 칸에 들어갈 말투로 작성해줘.
                - 위반 일시: {incident_datetime}
                - 위반 장소: {analysis_result.get("location", "")}
                - 위반 항목: {violation_type}
                - 차량 번호: {analysis_result.get("plate", "")}
                """

                # 함수 호출해서 초안 생성
                print(f"📝 신고 초안 생성 요청 중... (위반: {violation_type})")
                ai_description = llm_manager.get_report_draft(draft_prompt)
                print(f"✅ AI가 생성한 신고 초안: {ai_description[:30]}...")
            else:
                ai_description = "위반 사항 없음 또는 분석 실패"
            # ---------------- [추가된 코드 끝] ----------------

            # 3. 자바 서버로 보낼 최종 데이터(payload) 구성
            # (Java의 IncidentLogDTO와 매핑됩니다)
            payload = {
                "serialNo": serial_no,
                "videoUrl": video_url,
                "incidentDate": incident_date,
                "incidentTime": incident_time,
                "violationType": violation_type,
                "plateNo": analysis_result.get("plate", "-"),
                "location": analysis_result.get("location", ""),
                
                "aiDraft": ai_description  # <--- ★ 상세 내용(초안) 추가됨!
            }
            
            detection_logs.append(payload)

            # 4. Java(Spring) 서버로 결과 전송
            if USE_JAVA_SYNC:
                try:
                    requests.post(JAVA_SERVER_URL, json=payload, timeout=3)
                    print(f"📡 Java 서버 전송 완료: {JAVA_SERVER_URL}")
                except Exception as java_e:
                    print(f"⚠️ Java 서버 전송 실패: {java_e}")
            
            print(f"✅ 분석 및 전송 완료: {violation_type}")

            # 임시 파일 정리
            if os.path.exists(local_path): 
                os.remove(local_path)
            
        except Exception as e:
            print(f"❌ 전체 프로세스 에러: {e}")
        finally:
            if filename in processing_files: 
                processing_files.remove(filename)

ai_manager = AIService()