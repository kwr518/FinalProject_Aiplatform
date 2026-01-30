import os
import cv2
import numpy as np
import tensorflow as tf
import csv
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

# 번호판 인식 모듈
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
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        
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
            self.lpr_system = PlateRecognizerModule(YOLO_PATH)
        except:
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

                # 1. YOLO(.pt) 실시간 탐지 실행
                if self.obj_detector:
                    results = self.obj_detector(frame, conf=0.4, verbose=False)
                    for box in results[0].boxes:
                        name = self.obj_detector.names[int(box.cls[0])]
                        detected_items.add(name)

                # 프레임 전처리 (TF 모델용)
                all_frames.append(cv2.resize(frame, (128, 128)) / 255.0)
            
            cap.release()

            # 2. 위반 판단 (TensorFlow - .h5 모델)
            if len(all_frames) < SEQUENCE_LENGTH:
                return {"result": "분석 불가(영상 짧음)", "prob": 0, "plate": "-"}

            windows = [all_frames[i : i + SEQUENCE_LENGTH] for i in range(0, len(all_frames) - SEQUENCE_LENGTH + 1, STEP_SIZE)]
            predictions = self.model.predict(np.array(windows), batch_size=2, verbose=0)
            
            best_prob, best_class_idx, best_window_idx = 0, -1, -1
            for i, pred in enumerate(predictions):
                idx = np.argmax(pred)
                if pred[idx] > best_prob:
                    best_prob, best_class_idx, best_window_idx = pred[idx], idx, i

            # 3. 결과 정리 및 YOLO 데이터 합치기
            raw_label = CATEGORIES[best_class_idx] if best_class_idx != -1 else "정상 주행"
            obj_summary = ", ".join(list(detected_items)) if detected_items else "없음"
            
            # 🚀 [핵심 수정] 결과 문구에 YOLO 탐지 객체 정보를 포함시킵니다.
            final_display_result = f"{raw_label} ({obj_summary})"

            plate_text = "인식 불가"
            if self.lpr_system and best_window_idx != -1:
                plate_text = self.lpr_system.process_segment(local_path, best_window_idx * STEP_SIZE, SEQUENCE_LENGTH) or "인식 불가"

            return {
                "result": final_display_result, # 합쳐진 결과 전송
                "plate": plate_text,
                "location": "수원시 팔달구 매산로 1",
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "prob": round(float(best_prob * 100), 2),
                "info": f"YOLO 감지: {obj_summary}",
                "video_url": "" 
            }

        except Exception as e:
            print(f"❌ 로컬 분석 에러: {e}")
            return {"result": "에러 발생", "prob": 0, "plate": "Error"}

    def process_video_task(self, video_key):
        """S3 업로드 시 백그라운드 분석 태스크 (기존 유지)"""
        decoded_key = urllib.parse.unquote_plus(video_key)
        filename = os.path.basename(decoded_key)

        if filename in processing_files: return
        processing_files.add(filename)

        try:
            local_path = os.path.join(TEMP_VIDEO_DIR, filename)
            s3_manager.download_file(decoded_key, local_path)
            
            # analyze_local_video 로직과 동일하게 처리하도록 결과 호출
            payload = self.analyze_local_video(local_path)
            payload["video_url"] = s3_manager.get_presigned_url(decoded_key)
            
            detection_logs.append(payload)

            if USE_JAVA_SYNC:
                requests.post(JAVA_SERVER_URL, json=payload, timeout=3)
            
            print(f"✅ 분석 완료: {payload['result']}")

            if os.path.exists(local_path): os.remove(local_path)
            processing_files.remove(filename)

        except Exception as e:
            print(f"❌ 분석 에러: {e}")
            if filename in processing_files: processing_files.remove(filename)

ai_manager = AIService()