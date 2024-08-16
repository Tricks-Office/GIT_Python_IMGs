import tkinter as tk
import tkinter.ttk
from tkinter import filedialog
import pandas as pd
import os
import cv2
import numpy as np
from datetime import datetime
import shutil
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import joblib
import re

global model
global cnn_model

### 기준정보로 등록해야 하는 정보
## GUI 크기
Gsize = "630x450"

## 폴더 / 파일 리스트
# 각 행의 라벨 정보 List로 정리
label_f = ['Asian folder', 'English folder', 'Target folder','Result folder', 'Model file', 'OK folder', 'NG folder','Target folder', 'Result folder']
# 각 행에서 다루는 값이 폴더일때는 0, 파일일때는 1로 구분자
fileyn = [0, 0, 0, 0, 1, 0, 0, 0, 0]
# 폴더 / 파일 리스트 기준정보 불러오기 (기준정보 관리 파일명 : GUIMaster.csv)
# 기준정보 파일이 없을 경우 초기화
try : 
    df_FileFolder= pd.read_csv("Image_Judge_Master.csv")
except :
    d = {'Item' : ['파일 / 폴더 경로를 설정해 주세요'] * len(label_f)}
    df_FileFolder = pd.DataFrame(data=d)

## 파라미터 텍스트 상자 리스트
label_para = ['Spec', 'Left', 'Right', 'Top', 'Bottom', 'Left', 'Right', 'Top', 'Bottom']
# 폴더 / 파일 리스트 기준정보 불러오기 (기준정보 관리 파일명 : Para.csv)
# 기준정보 파일이 없을 경우 초기화
try : 
    dfP= pd.read_csv("Image_Judge_Para.csv")
except :
    d = {'Item' : ['값을 입력해 주세요'] * len(label_para)}
    dfP = pd.DataFrame(data=d)

### 주요 함수
## 개발된 함수 추가
def temp(test):
    return

# 한글을 영문으로 변환하는 함수
def korean_to_english(text):
    # 로마자 변환 (ASCII 문자로 변환)
    converted_text = unidecode(text)
    return converted_text

# 디렉토리 내의 모든 파일 처리
def process_directory(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for root, dirs, files in os.walk(source_dir):
        for file_name in files:
            # 한글이 포함된 파일 이름을 영문으로 변환
            new_file_name = korean_to_english(file_name)
            # 새로운 파일 경로
            source_file = os.path.join(root, file_name)
            new_root = root.replace(source_dir, destination_dir)
            new_file_path = os.path.join(new_root, new_file_name)
            
            # 새 디렉토리 경로가 존재하지 않으면 생성
            if not os.path.exists(new_root):
                os.makedirs(new_root)
            
            # 파일 이동
            shutil.copy(source_file, new_file_path)

# 한글 파일 to 영어 파일 실행
def ko2en():
    source_dir = df_FileFolder.Item[0]
    destination_dir = df_FileFolder.Item[1]
    process_directory(source_dir, destination_dir)

# Image Simple Judgement
def simple_judge(judgetype):
    update_Para()
    target_folder = df_FileFolder.Item[2]
    result_folder = df_FileFolder.Item[3]
    SJ_spec = int(dfP.Item[0])
    SJ_left = int(dfP.Item[1])
    SJ_right = int(dfP.Item[2])
    SJ_top = int(dfP.Item[3])
    SJ_bottom = int(dfP.Item[4])

    for filename in os.listdir(target_folder):
        file = os.path.join(target_folder, filename)
        savefilename = os.path.join(result_folder, filename)
        # 이미지 로드
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        cutted_image = image[SJ_top:SJ_bottom, SJ_left:SJ_right]
        if image is None:
            print("이미지를 불러오지 못했습니다.")
            return

        # 이미지 흐리게 만들기
        blurred_img = cv2.blur(cutted_image,(5,5))

        match judgetype:
            case 0 :
                # 이미지 이진화 (임계값 : spec 의 70% 값 사용, 필요시 조정 가능)
                boundary_spec = SJ_spec * 0.7
                _, binary_image = cv2.threshold(blurred_img, boundary_spec, 255, cv2.THRESH_BINARY)
                # 검은색 영역 찾기 (명도 값이 0인 부분)
                black_area = cutted_image[binary_image == 0]
                # 검은색 영역 내 명도가 spec을 초과하는지 확인
                if np.any(black_area > int(SJ_spec)):
                    shutil.copy(file, savefilename)
            case 1 : 
                # 이미지 이진화 (임계값 : 255 - spec 보다 30% 높은 값 사용, 필요시 조정 가능)
                boundary_spec = SJ_spec + ((255 - SJ_spec) * 0.3)
                _, binary_image = cv2.threshold(blurred_img, boundary_spec, 255, cv2.THRESH_BINARY)
                # 검은색 영역 찾기 (명도 값이 255인 부분)
                white_area = cutted_image[binary_image == 255]
                # 검은색 영역 내 명도가 spec 미만인지 확인
                if np.any(white_area < int(SJ_spec)):
                    shutil.copy(file, savefilename)


def load_images_from_folder(folder, label):
    images = []
    labels = []
    ML_left = int(dfP.Item[5])
    ML_right = int(dfP.Item[6])
    ML_top = int(dfP.Item[7])
    ML_bottom = int(dfP.Item[8])
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cutted_image = img[ML_top:ML_bottom, ML_left:ML_right]
        
        if cutted_image is not None:
            cutted_image = cv2.resize(cutted_image, (128, 128))  # 이미지 크기 통일 (예: 128x128)
            images.append(cutted_image.flatten())  # SVM 및 로지스틱 회귀 모델용
            labels.append(label)
    return images, labels

def load_dataset():
    ok_folder = df_FileFolder.Item[5]
    ng_folder = df_FileFolder.Item[6]
    ok_images, ok_labels = load_images_from_folder(ok_folder, 0)  # OK 폴더의 이미지들은 라벨 0
    ng_images, ng_labels = load_images_from_folder(ng_folder, 1)  # NG 폴더의 이미지들은 라벨 1

    images = np.array(ok_images + ng_images)
    labels = np.array(ok_labels + ng_labels)

    return train_test_split(images, labels, test_size=0.3, random_state=42)

def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_no):
    global model
    match model_no:
        case 0:
            # 로지스틱 회귀 모델 학습 및 검증
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            lr_predictions = model.predict(X_test)
            print("\nLogistic Regression Model")
            print("Accuracy:", accuracy_score(y_test, lr_predictions))
            print("Classification Report:\n", classification_report(y_test, lr_predictions))
        case 1:
            # SVM 모델 학습 및 검증
            model = SVC()
            model.fit(X_train, y_train)
            svm_predictions = model.predict(X_test)
            print("SVM Model")
            print("Accuracy:", accuracy_score(y_test, svm_predictions))
            print("Classification Report:\n", classification_report(y_test, svm_predictions))

def load_images_for_cnn(folder, label):
    images = []
    labels = []
    ML_left = int(dfP.Item[5])
    ML_right = int(dfP.Item[6])
    ML_top = int(dfP.Item[7])
    ML_bottom = int(dfP.Item[8])
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cutted_image = img[ML_top:ML_bottom, ML_left:ML_right]
        if cutted_image is not None:
            cutted_image = cv2.resize(cutted_image, (128, 128))  # 이미지 크기 통일 (예: 128x128)
            images.append(cutted_image)
            labels.append(label)
    return images, labels

def load_dataset_for_cnn():
    ok_folder = df_FileFolder.Item[5]
    ng_folder = df_FileFolder.Item[6]
    ok_images, ok_labels = load_images_for_cnn(ok_folder, 0)
    ng_images, ng_labels = load_images_for_cnn(ng_folder, 1)

    images = np.array(ok_images + ng_images)
    labels = np.array(ok_labels + ng_labels)

    images = images.reshape(images.shape[0], 128, 128, 1)
    labels = to_categorical(labels, 2)

    return train_test_split(images, labels, test_size=0.3, random_state=42)

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_cnn_model(X_train, X_test, y_train, y_test):
    global model
    model = create_cnn_model()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("\nCNN Model")
    print("Test accuracy:", test_acc)

def save_models(model_no):
    global model
    # 현재시간 가져와 결과 파일만들기
    now_day = datetime.today().strftime('%Y%m%d')
    
    match model_no:
        case 0:
            model_path = now_day + '_lr_model.pkl'
            joblib.dump(model, model_path)
        case 1:
            model_path = now_day + '_svm_model.pkl'
            joblib.dump(model, model_path)
            
        case 2:
            model_path = now_day + '_cnn_model.h5'
            model.save(model_path)
                        
    lbPath[4].delete('1.0', tk.END)
    lbPath[4].insert(tk.INSERT, chars=model_path)
    update_Master(4, model_path)

def load_models():
    global model
    filename = df_FileFolder.Item[4]
    if 'lr' in filename:
        model = joblib.load(filename)
        MLJ_var.set(0)
    elif 'svm' in filename:
        model = joblib.load(filename)
        MLJ_var.set(1)
    elif 'cnn' in filename:
        model = load_model(filename)
        MLJ_var.set(2)
    

def training_model(model_no):
    update_Para()
    match model_no:
        case 0 | 1:
            X_train, X_test, y_train, y_test = load_dataset()
            train_and_evaluate_models(X_train, X_test, y_train, y_test, model_no)            
        case 2:
            X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = load_dataset_for_cnn()
            train_and_evaluate_cnn_model(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn)
    save_models(model_no)

def run_model(model_no):
    update_Para()
    target_folder = df_FileFolder.Item[7]
    result_folder = df_FileFolder.Item[8]
    ML_left = int(dfP.Item[5])
    ML_right = int(dfP.Item[6])
    ML_top = int(dfP.Item[7])
    ML_bottom = int(dfP.Item[8])

    for filename in os.listdir(target_folder):
        img_path = os.path.join(target_folder, filename)
        result_path = os.path.join(result_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[ML_top:ML_bottom, ML_left:ML_right]
        if img is not None:
            img_resized = cv2.resize(img, (128, 128))
            img_flat = img_resized.flatten().reshape(1, -1)
            img_cnn = img_resized.reshape(1, 128, 128, 1)

            match model_no:
                case 0|1:
                    model_prediction = model.predict(img_flat)
                case 2:
                    model_prediction = np.argmax(model.predict(img_cnn), axis=1)    
            if model_prediction > 0.5:  
                shutil.copy(img_path, result_path)

### GUI용 함수
# GUIMaster Data 업데이트 : 
def update_Master(idx, var):
    df_FileFolder.Item[idx] = var
    df_FileFolder.to_csv('Image_Judge_Master.csv', index=False)

# Para Data 업데이트 :
def update_Para():
    for i in range(len(label_para)):
        dfP.Item[i] = txtPara[i].get("1.0",tk.END)
    dfP.to_csv('Image_Judge_Para.csv', index=False)

# 폴더/파일 경로 바꾸는 버튼을 눌렀을때 업데이트
def onClick(i, fileYN):
    # 폴더 경로 바꾸는 로직 (fineYN = 0 일때)
    if fileYN == 0:
        folder_selected = filedialog.askdirectory()
        var = folder_selected
    # 파일 경로 바꾸는 로직 (fineYN = 0 이 아닐때)
    else:
        folder_selected = filedialog.askopenfile()
        var = folder_selected.name

    lbPath[i].delete('1.0', tk.END)
    lbPath[i].insert(tk.INSERT, chars=var)
    update_Master(i,var)
    
## Main Code
# GUI 구성
win = tk.Tk()
win.geometry(Gsize)
win.title('Python image judge')

# Frame 설정하기
notebook = tkinter.ttk.Notebook(win)
notebook.pack()

frameChangePath = tk.Frame(win, pady=5, padx = 5)
notebook.add(frameChangePath, text = "Change Path")
frameSimpleJudge = tk.Frame(win, pady=5, padx = 5)
notebook.add(frameSimpleJudge, text = "Simple Judge")
frameMLJudge = tk.Frame(win, pady=5, padx = 5)
notebook.add(frameMLJudge, text = "ML Judge")

frameCP_F = tk.Frame(frameChangePath, pady=5, padx = 5)
frameCP_F.grid(row=0, column=0, sticky= "ew", padx=5,pady=5)
frameCP_B = tk.Frame(frameChangePath, pady=5, padx = 5)
frameCP_B.grid(row=1, column=0, sticky= "ew", padx=5, pady=5)

frameSJ_F = tk.Frame(frameSimpleJudge, pady=5, padx = 5)
frameSJ_F.grid(row=0, column=0, sticky= "ew", padx=5, pady=5)
frameSJ_R = tk.Frame(frameSimpleJudge, pady=5, padx = 5)
frameSJ_R.grid(row=1, column=0, sticky= "ew", padx=5, pady=5)
frameSJ_P = tk.Frame(frameSimpleJudge, pady=5, padx = 5)
frameSJ_P.grid(row=2, column=0, sticky= "ew", padx=5, pady=5)
frameSJ_B = tk.Frame(frameSimpleJudge, pady=5, padx = 5)
frameSJ_B.grid(row=3, column=0, sticky= "ew", padx=5, pady=5)

frameMLJ_F = tk.Frame(frameMLJudge, pady=5, padx = 5)
frameMLJ_F.grid(row=0, column=0, sticky= "ew", padx=5, pady=5)
frameMLJ_R = tk.Frame(frameMLJudge, pady=5, padx = 5)
frameMLJ_R.grid(row=1, column=0, sticky= "ew", padx=5, pady=5)
frameMLJ_P = tk.Frame(frameMLJudge, pady=5, padx = 5)
frameMLJ_P.grid(row=2, column=0, sticky= "ew", padx=5, pady=5)
frameMLJ_B = tk.Frame(frameMLJudge, pady=5, padx = 5)
frameMLJ_B.grid(row=3, column=0, sticky= "ew", padx=5, pady=5)

# 파일 / 폴더 경로 설정 GUI
lbFame = []
lbPath = []
btnPath =[]

for i,x in enumerate(label_f):
    match i:
        case 0 | 1 :
            lbFame.append(tk.Label(frameCP_F, text=x, width=15,padx =5, pady = 5))
            lbPath.append(tk.Text(frameCP_F, width = 50, height = 1, padx =5, pady = 5, background='lightgrey'))
            btnPath.append(tk.Button(frameCP_F, text="Change Path", width=10, padx =5, pady = 5, command=lambda i=i: onClick(i,fileyn[i])))
        case 2 | 3 :
            lbFame.append(tk.Label(frameSJ_F, text=x, width=15, padx =5, pady = 5))
            lbPath.append(tk.Text(frameSJ_F, width = 50, height = 1, padx =5, pady = 5, background='lightgrey'))
            btnPath.append(tk.Button(frameSJ_F, text="Change Path", width=10, padx =5, pady = 5, command=lambda i=i: onClick(i,fileyn[i])))
        case 4 | 5 | 6 | 7 | 8  :
            lbFame.append(tk.Label(frameMLJ_F, text=x, width=15, padx =5, pady = 5))
            lbPath.append(tk.Text(frameMLJ_F, width = 50, height = 1, padx =5, pady = 5, background='lightgrey'))
            btnPath.append(tk.Button(frameMLJ_F, text="Change Path", width=10, padx =5, pady = 5, command=lambda i=i: onClick(i,fileyn[i])))

    # 폴더/파일 이름 초기값 넣기
    lbPath[i].insert(tk.INSERT, chars=df_FileFolder.Item[i])

    lbFame[i].grid(row=i, column=0, padx =5, sticky=tk.W)
    lbPath[i].grid(row=i, column=1, padx =5, sticky=tk.W)
    btnPath[i].grid(row=i, column=2, padx =5, sticky=tk.W)

# Parameter 설정 GUI
lbPara = []
txtPara = []

for i,x in enumerate(label_para):
    match i:
        case 0 :
            lbPara.append(tk.Label(frameSJ_P, text=x, width=10))
            txtPara.append(tk.Text(frameSJ_P, width = 25, height = 1, padx =5, pady = 5))
            lbPara[i].grid(row=i, column=0, sticky=tk.W)
            txtPara[i].grid(row=i, column=1, sticky=tk.W)
        case 1 | 2 | 3 | 4 :
            lbPara.append(tk.Label(frameSJ_P, text=x, width=10))
            txtPara.append(tk.Text(frameSJ_P, width = 25, height = 1, padx =5, pady = 5))
            lbPara[i].grid(row=((i-1) // 2) + 1, column = ((i-1) % 2 )* 2 , sticky=tk.W)
            txtPara[i].grid(row=((i-1) // 2) + 1, column=((i-1) % 2 ) * 2 + 1, sticky=tk.W)
        case 5 | 6 | 7 | 8 :
            lbPara.append(tk.Label(frameMLJ_P, text=x, width=10))
            txtPara.append(tk.Text(frameMLJ_P, width = 25, height = 1, padx =5, pady = 5))
            lbPara[i].grid(row=((i-1) // 2) + 1, column = ((i-1) % 2 )* 2 , sticky=tk.W)
            txtPara[i].grid(row=((i-1) // 2) + 1, column=((i-1) % 2 ) * 2 + 1, sticky=tk.W)

    # 파라미터 초기값 넣기
    txtPara[i].insert(tk.INSERT, chars=dfP.Item[i])
    
# Simple Judge Radio 상자 GUI
SJ_var = tk.IntVar()
SJ_var.set(0)

SJ_List = ['White dot in Black area','Black dot in White area' ]
SJ_radio = []
for i,x in enumerate(SJ_List):
    SJ_radio.append(tk.Radiobutton(frameSJ_R, text=x, variable=SJ_var, width = 35, padx=1, value=i))
    SJ_radio[i].grid(row=0, column=i, sticky=tk.W)

# ML Judge Radio 상자 GUI
MLJ_var = tk.IntVar()
MLJ_var.set(0)
MLJ_List = ['LR', 'SVM', 'CNN' ]
MLJ_radio = []
for i,x in enumerate(MLJ_List):
    MLJ_radio.append(tk.Radiobutton(frameMLJ_R, text=x, variable=MLJ_var, width = 23, padx=1, value=i))
    MLJ_radio[i].grid(row=0, column=i, sticky=tk.W)

# Path Change Button GUI
button_path = tk.Button(frameCP_B, text="Apply Change", width = 30, padx=5, command=lambda: ko2en())
# button_path.grid(row=0, column=0, sticky=tk.E)
button_path.pack()

# Simple Judge Button GUI
button_simple = tk.Button(frameSJ_B, text="Run Simple Judge", width = 30, padx=5, command=lambda: simple_judge(SJ_var.get()))
button_simple.pack()

# ML Judge Button GUI
button_ML_train = tk.Button(frameMLJ_B, text="ML Training", width = 30, padx=5, command=lambda: training_model(MLJ_var.get()))
button_ML_train.pack(pady = 5)
button_ML_loadmodel = tk.Button(frameMLJ_B, text="Load ML Model", width = 30, padx=5, command=lambda: load_models())
button_ML_loadmodel.pack(pady = 5)
button_ML_predict = tk.Button(frameMLJ_B, text="Run ML Judge", width = 30, padx=5, command=lambda: run_model(MLJ_var.get()))
button_ML_predict.pack()

win.mainloop()