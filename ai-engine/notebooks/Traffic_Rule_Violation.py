#!/usr/bin/env python
# coding: utf-8

# ## LRCN 분류

# # Modules

# In[1]:

import os
import cv2
import math
import random
import datetime
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model


# In[ ]:


# seed 27번 고정 
seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)


# In[ ]:


# signal: 신호등 상태 인식, middleLine: 중앙선 침범, whiteLine: 차선 위반
all_classes = ['Signal', 'middleLine', 'whiteLine']
categories = ['신호위반','중앙선침범','진로변경위반']    # 모델이 분류할 항목
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64                 # 가로 x 세로 64px 리사이징
SEQUENCE_LENGTH = 25                                # 하나의 영상 샘플당 프레임 개수


# In[ ]:


# path 경로에 폴더 없을 경우 디렉토리 생성
def createDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

# 폴더 내 지정된 프레임들을 모아 모델 입력용 시퀸스 데이터로 변환
def frame_extraction(folder_path):
    frame_list = []
    
    # [Point 1] 파일명 정렬 (필수!)
    try:
        file_names = sorted(os.listdir(folder_path))
    except FileNotFoundError:
        print(f"❌ 경로 없음: {folder_path}")
        return []
    
    file_paths = [os.path.join(folder_path, i) for i in file_names]
    
    for file in file_paths:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):    # 이미지 확장자 필터링
            continue
            
        try:
            img_array = np.fromfile(file, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # [Point 2] 안전장치
            if frame is None: continue
            
            resized = cv2.resize(frame, (IMAGE_HEIGHT , IMAGE_WIDTH))
            normalized = resized / 255.0
            frame_list.append(normalized)
            
            if len(frame_list) == SEQUENCE_LENGTH:
                break
        except Exception as e:
            print(f"이미지 로드 에러: {e}")

    # [Point 3] 패딩 (프레임 부족 시 채움)
    if 0 < len(frame_list) < SEQUENCE_LENGTH:
        while len(frame_list) < SEQUENCE_LENGTH:
            frame_list.append(frame_list[len(frame_list) % len(frame_list)])
            
    return frame_list


# In[ ]:
def get_data(paths, labels):    
    for folder_path, label_index in zip(paths,labels):  # 폴더 경로 리스트와 정답 라벨 리스트 쌍으로 묶기
        feature = frame_extraction(folder_path) # frame_extraction 호출 후 25장 정규화 프레임 가져오기
        feature = np.array(feature) # 리스트 데이터를 모아 넘파이 배열로 수정

        label = np.array([label_index]) # (피처, 라벨) 쌍을 모델 학습 루프에 전달
        yield (feature, label)      # yield 사용 시 현재 처리중인 영상 한개만 메모리에 유지하여 성능 방어


# In[ ]:


def create_paths(data_type):
    labels = []
    paths = []
    for label in categories:
        print("진행중:", label)
        
        path = f"/mnt/traffic/교통데이터/{data_type}이미지데이터/{label}"
        for root, directories, files in os.walk(path):  # 현탐색 중인 경로 내 하위 폴더까지 리스트 찾기
            for file in files:  
                if file.split('.')[-1] =='jpg' or file.split('.')[-1]=='jpeg':  # jpg, jpeg 이미지인지 확인
                    
                    if root not in paths:   # 폴더 경로가 이미 리스트에 등록되어 있지 않은 경우 데이터 경로 추가
                        paths.append(root)  
                        labels.append(categories.index(label))  # 위반 항목 숫자로 변환해 정답 리스트에 넣기
    print('=========================')
    return paths, labels


# In[ ]:

class Dataloader(Sequence):
    def __init__(self, x_set, y_set, batch_size, shuffle=False):
        self.x, self.y = x_set, y_set   # 경로 리스트 및 정답 라벨 리스트 저장
        self.batch_size = batch_size    # 한번에 학습할 데이터 양
        self.shuffle=shuffle            # 한 에폭이 끝날때마다 데이터 순서 섞기 
        self.on_epoch_end()             # 초기화 시점에 한 번 호출하여 데이터 인덱스 생성

    # 전체 개수를 배치 사이즈로
    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size) # 소수점일 경우 올림하여 자투리 데이터 포함

	# batch 단위로 직접 묶어줘야 함
    def __getitem__(self, idx):
				# sampler의 역할(index를 batch_size만큼 sampling해줌)
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [frame_extraction(self.x[i]) for i in indices]    # 추출된 인덱스를 호출해 이미지를 읽어와 25장의 시퀸스 데이터로 변환
        batch_y = [self.y[i] for i in indices]  # 해당 영상들의 정답 라벨들을 리스트로 모으기
    
        # 0,1,2 본류 모델 이해를 위한 원핫 인코딩으로 변경(신호위반, 중앙선침범, 진로변경위반)
        return np.asarray(batch_x), to_categorical(np.array(batch_y), num_classes=3)    

    # epoch이 끝날때마다 실행
    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))   # 데이터 순서대로 인덱스 번호 다시 생성
        if self.shuffle == True:    # 인덱스 순서 무작위로 섞어 데이터 순서를 외우는 과적합 방지
            np.random.shuffle(self.indices)


# In[9]:

train_paths, train_labels = create_paths('')    # 학습용 데이터 준비
val_paths, val_labels = create_paths('val_')    # 폴더 경로 탐색 후 학습 중 성능을 체크할 검증 데이터 준비


# In[ ]:

test_paths, test_labels = create_paths('test_') # 폴더 경로 탐색 및 죄종 성능 평가용 데이터 준비

# In[11]:

# Dataloader 함수에 호출
train_dataset = Dataloader(train_paths, train_labels, 24, shuffle=True)
val_dataset =  Dataloader(val_paths, val_labels, 24)
test_dataset =  Dataloader(test_paths, test_labels, 24)


# In[12]:


# LRCN : 이미지의 특징을 추출하는 CNN과 시간 흐름을 분석하는 LSTM의 결합형태
categories = ['신호위반','중앙선침범','진로변경위반']
def create_LRCN_model():
    model = Sequential()

    # TimeDistributed : 입력받은 25개의 프레임 각각에 동일한 CNN 연산 적용
    # Conv2D : 16개의 필터를 사용해 기초적인 특징(선, 면 등)을 찾기
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) # MaxPooling(4, 4) : 이미지 크기를 1/4로 줄여 중요한 정보만 남기기
    model.add(TimeDistributed(Dropout(0.25))) # Dropout(0.25) : 학습 시 노드의 25% 무작위로 꺼서 과적합 방지
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(TimeDistributed(Flatten()))   # CNN을 통해 나온 2차원 특징 맵을 1차원 벡터로 펼치기               
    model.add(LSTM(32)) # 25개 프레임의 흐름을 기억하며 분석(차량 궤적 중앙선 침범, 신호 위반 등 동적 움직임 판단)
    model.add(Dense(len(categories), activation = 'softmax')) # 최종 출력 노드 3개 및, 각 위반 항목에 해당할 확률은 0 ~ 1사이로 출력하며 세 항목의 합은 1

    
    model.summary()     # 모델 요약 디스플레이
    return model        # LRCN 모델 리턴
LRCN_model = create_LRCN_model()


# In[17]:


dir_name = 'snap_shot'

# 학습 과정 기록 및 시각화(TensorBoard)
def make_Tensorboard_dir(dir_name): # 실행 시간 별 폴더 제작
    root = os.path.join(os.curdir, dir_name)    # snap_shot 이름의 메인 폴더 경로 지정
    sub_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 하위 폴더로 'YMD-HMS'형태로 폴더 명 생성
    
    # 폴더 없을 경우 실제 생성
    if not os.path.exists(root):    
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, sub_dir)):
        os.mkdir(os.path.join(root, sub_dir))
    return os.path.join(root, sub_dir)

# 텐서보드 콜백 설정
TB_log_dir = make_Tensorboard_dir(dir_name) # 위 함수를 실행해 실제 로그가 저장될 최종 경로 설정

# model.fit의 callbacks 리스트에 넣으면 학습 과정 중 발생하는 모든 지표 해당 파일에 기록됨
TensorB = tf.keras.callbacks.TensorBoard(log_dir = TB_log_dir)  


# In[21]:

# 조기 종료 설정(EarlyStopping)
early_stopping_callback = EarlyStopping(monitor = 'val_loss',  # 검증 데이터 손실값 관찰
                                        patience = 15,  # 15 Epoch 동안 향상 없을 경우 학습 중단
                                        mode = 'min',   # 지표가 줄어드는 것이 멈출 때 중단
                                        restore_best_weights = True)    # 학습 중단 시 Loss가 가장 낮았던 시점의 가중치로 모델 선택
 

LRCN_model.compile(loss = 'categorical_crossentropy', # 다중 클래스 분류에 사용하는 표준 손실 함수
                   optimizer = 'Adam', # 가장 널리 쓰이는 학습 최적화 알고리즘
                   metrics = ["accuracy"])  # 터미널에 정확도 출력
 
# Start training the model.

with tf.Graph().as_default():
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)    # 학습에 필요한 만큼만 메모리 할당
with tf.device("/device:GPU:1"):    # 컴퓨터에 설치된 GPU 인덱스 선택하여 수행
    LRCN_model_training_history2 = LRCN_model.fit(train_dataset, # Dataloader로 만든 데이터 공급
                                                  epochs = 500,  
                                                  workers=4 ,    # CPU 4EA 병렬 처리
                                                  shuffle = True,   # 셔플
                                                  validation_data= val_dataset, # 학습 중간마다 검증 데이터를 넣어 처음 보는 영상도 잘맞히는지 성능 체크
                                                  callbacks = [early_stopping_callback, TensorB])   # 검증 손실이 줄어들지 않은 경우 조기 종료 및 로그 저장


# In[23]:


def plot_metric(model_training_history, # 학습 중 기록된 loss, accuracy 데이터가 담긴 객체
                metric_name_1,  # 그래프에 출력할 데이터의 이름
                metric_name_2,   
                plot_name): # 그래프 제목
   
    # hisory.history 딕셔너리에서 원하는 지표의 숫자 리스트 꺼내오기
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    
    # 그래프 X축으로 학습된 횟수만큼 순서대로 숫자 생성
    epochs = range(len(metric_value_1))

    # Blue, Red 선 그리기
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # 학습용 및 검증용 설명 텍스트로 구분
    plt.legend()


# In[24]:
plot_metric(LRCN_model_training_history2, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# In[25]:
plot_metric(LRCN_model_training_history2, 'accuracy', 'val_accuracy', 'Total Accuracy vs Total Validation Accuracy')

# In[26]:
# 학습에 사용되지 않은 새로운 데이터를 넣어 모델 검증 진행
model_evaluation_history = LRCN_model.evaluate(test_dataset)

# 모델 저장
# In[29]:
# 평가 결과에서 손실값과 정확도 추출
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
 
# datetime 모듈로 현재 시간을 초 단위까지 문자열로 설정
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# 파일 이름안에 저장 시간, 손실도, 정확도 전부 넣기
model_file_name = f'LRCN_model___Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'
save_dir = './lstm_model'
createDirectory(save_dir)

# 모델 구조, 가중치, 최적화 설정 전부 넣기
LRCN_model.save(os.path.join(save_dir, model_file_name))

# In[2]:

# 모델 불러오기
LRCN_model = tf.keras.models.load_model('./lstm_model/LRCN_model___Date_Time_2023_03_01__04_35_27___Loss_0.01098685897886753___Accuracy_0.9961240291595459.h5')

# # 폴더별 f1 score 구하기

# In[12]:
# 폴더별 f1_score 측정

# In[46]:


# =========================================================
# 2. 통합 평가 및 로그 저장 함수 (In[46] + In[47] 대체)
# =========================================================
def evaluate_and_log(test_root_folder, save_path='/mnt/traffic/lstm_f1_score.txt'):
    y_true = [] # 실제 정답
    y_pred = [] # 모델 예측
    
    print(f"🚀 분석 시작: {test_root_folder}")
    
    # 3가지 클래스 폴더를 모두 순회
    for label_idx, label_name in enumerate(categories):
        target_dir = os.path.join(test_root_folder, label_name)
        if not os.path.exists(target_dir):
            print(f"⚠️ 경로 없음 패스: {target_dir}")
            continue
            
        # 해당 클래스 폴더 내의 모든 영상 폴더 탐색
        video_folders = []
        for root, dirs, files in os.walk(target_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                video_folders.append(root)
        
        print(f"   - [{label_name}] 영상 {len(video_folders)}개 분석 중...")
        
        for folder in video_folders:
            frames = frame_extraction(folder)
            if not frames: continue
            
            # 모델 예측
            pred_prob = LRCN_model.predict(np.expand_dims(frames, axis=0), verbose=0)[0]
            pred_idx = np.argmax(pred_prob)
            
            # 결과 수집
            y_true.append(label_idx)
            y_pred.append(pred_idx)

    print("✅ 분석 완료! 결과 집계 중...\n")

    # -----------------------------------------------------
    # 3. 결과 리포트 생성 및 파일 저장
    # -----------------------------------------------------
    
    # Sklearn이 제공하는 정확한 성능 보고서 (FP, FN 완벽 계산됨)
    report = classification_report(y_true, y_pred, target_names=categories, digits=4)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # 화면 출력
    print("="*60)
    print(" [ 최종 성능 평가 결과 ] ")
    print("="*60)
    print(report)
    print("\n[혼동 행렬 (Confusion Matrix)]")
    print(conf_matrix)
    print("="*60)
    
    # 파일 저장 (기존 In[47]의 목적 달성)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write(f"테스트 경로: {test_root_folder}\n")
        f.write("="*60 + "\n\n")
        f.write(report)
        f.write("\n\n[혼동 행렬]\n")
        f.write(str(conf_matrix))
        
    print(f"📁 결과가 파일에 저장되었습니다: {save_path}")

# =========================================================
# 3. 실행 (여기를 수정하세요)
# =========================================================

# 검증하고 싶은 데이터셋의 최상위 경로 (val 또는 test)
# 예: /mnt/traffic/교통데이터/lstm_val_이미지데이터
TEST_DATA_PATH = "/mnt/traffic/교통데이터/lstm_val_이미지데이터"

# 함수 실행 (이거 한 줄이면 끝납니다!)
evaluate_and_log(TEST_DATA_PATH)

# In[40]:
LRCN_model = tf.keras.models.load_model('./lstm_model/LRCN_model___Date_Time_2023_03_01__04_35_27___Loss_0.01098685897886753___Accuracy_0.9961240291595459.h5')

# In[45]:

def classifier(folder_path):
    categories = ['신호위반', '중앙선침범','진로변경위반']
    frame_list = frame_extraction(folder_path)
    predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frame_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    return print(f"탐지된 결과: {categories[predicted_label]}")

def createDirectory(path):
    if not os.path.exists(path):
        os.mkdir(path)

# # ======신호위반=======

# In[46]:

path = "/mnt/traffic/교통데이터/test_이미지데이터/신호위반/적색신호시직진/20230225_적색신호시직진_0000000007/"

path_ = path
fig = plt.figure(figsize=(10,10)) # rows*cols 행렬의 i번째 subplot 생성
rows = 5
cols = 5
i = 1
 
xlabels = [f"{x}" if x!=0 else 'xlabel' for x in range(26) ]


for filename in sorted(os.listdir(path_))[:25]:
    filename = os.path.join(path_, filename)
    img = cv2.imread(filename)
    
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(xlabels[i])
    ax.set_xticks([]), ax.set_yticks([])
    i += 1

plt.show()
classifier(path)


# In[58]:
# In[48]:
# In[49]:
# # ======중앙선침범=======

# In[50]:

path = "/mnt/traffic/교통데이터/test_이미지데이터/중앙선침범/중앙선주황색실선위반/20230227_중앙선주황색실선위반_0000000497/"

path_ = path
fig = plt.figure(figsize=(10,10)) # rows*cols 행렬의 i번째 subplot 생성
rows = 5
cols = 5
i = 1
 
xlabels = [f"{x}" if x!=0 else 'xlabel' for x in range(26) ]


for filename in sorted(os.listdir(path_))[:25]:
    filename = os.path.join(path_, filename)
    img = cv2.imread(filename)
    
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(xlabels[i])
    ax.set_xticks([]), ax.set_yticks([])
    i += 1

plt.show()
classifier(path)


# In[51]:
# In[52]:

# # ======진로변경위반======

# In[53]:
# In[54]:
# In[55]:
path = "/mnt/traffic/교통데이터/test_이미지데이터/진로변경위반/일반도로진로변경위반/20230227_일반도로진로변경위반_0000000496/"
path_ = path
fig = plt.figure(figsize=(10,10)) # rows*cols 행렬의 i번째 subplot 생성
rows = 5
cols = 5
i = 1
 
xlabels = [f"{x}" if x!=0 else 'xlabel' for x in range(26) ]

for filename in sorted(os.listdir(path_))[:25]:
    filename = os.path.join(path_, filename)
    img = cv2.imread(filename)
    
    ax = fig.add_subplot(rows, cols, i)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xlabel(xlabels[i])
    ax.set_xticks([]), ax.set_yticks([])
    i += 1

plt.show()
classifier(path)
