import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from datetime import datetime

### 기준정보로 등록해야 하는 정보
## GUI 크기
Gsize = "620x480"

## 폴더 / 파일 리스트
# 각 행의 라벨 정보 List로 정리
label_f = ['Target folder', 'Result folder', 'Training folder','Model file','Test Image']
# 각 행에서 다루는 값이 폴더일때는 0, 파일일때는 1로 구분자
fileyn = [0, 0, 0, 1, 1]
# 폴더 / 파일 리스트 기준정보 불러오기 (기준정보 관리 파일명 : GUIMaster.csv)
# 기준정보 파일이 없을 경우 초기화
try : 
    df_FileFolder= pd.read_csv("GUIMaster.csv")
except :
    d = {'Item' : ['파일 / 폴더 경로를 설정해 주세요'] * len(label_f)}
    df_FileFolder = pd.DataFrame(data=d)

### 주요 함수
## 개발된 함수 추가
def apply_filter(filter_no):
    target_folder = df_FileFolder.Item[0]
    output_image_folder = df_FileFolder.Item[1]
    
    # 이미지 로드
    for file in os.listdir(target_folder):
        input_image_path = os.path.join(target_folder, file)
        output_image_path = os.path.join(output_image_folder, file)
        
        img = cv2.imread(input_image_path, cv2.IMREAD_COLOR)

        # # 원본 이미지 출력
        # plt.figure(figsize=(10, 10))
        # plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        match filter_no:
            case 0 :
                filter_result = cv2.GaussianBlur(img, (5, 5), 0)
                
            case 1 :
                filter_result = cv2.medianBlur(img, 5)

            case 2 :
                filter_result = cv2.bilateralFilter(img, 9, 75, 75)
                
        cv2.imwrite(output_image_path, filter_result)

def apply_edge(edge_no):
        
    # 폴더 경로 설정
    target_folder = df_FileFolder.Item[0]
    output_image_folder = df_FileFolder.Item[1]

    for file in os.listdir(target_folder):
        
        # 이미지 로드
        input_image_path = os.path.join(target_folder, file)
        output_image_path = os.path.join(output_image_folder, file)

        img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        match edge_no:
            case 0 :
                # Canny 엣지 검출
                result_image = cv2.Canny(img, 100, 200)
            case 1 : 
                ## Contour Detection
                # 이진화
                ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

                # 윤곽선 찾기
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # 새로운 이미지에 윤곽선 그리기
                result_image = np.zeros_like(img)
                cv2.drawContours(result_image, contours, -1, (255, 255, 255), 1)

        # 결과 저장
        cv2.imwrite(output_image_path, result_image)

def apply_of(outfocus_no):
    # YOLO 모델 파일 및 구성 파일 설정
    yolo_cfg = 'yolov4.cfg'
    yolo_weights = 'yolov4.weights'
    yolo_names = 'coco.names'

    # COCO 클래스 이름 로드
    with open(yolo_names, 'r') as f:
        classes = f.read().splitlines()

    # 네트워크 모델 불러오기
    net = cv2.dnn.readNetFromDarknet(yolo_cfg, yolo_weights)
    layer_names = net.getLayerNames()

    # output_layers 얻기
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # 폴더 정보 로드
    input_image_folder = df_FileFolder.Item[0]
    output_image_folder = df_FileFolder.Item[1]

    # for 반복문
    for file in os.listdir(input_image_folder):
        # 이미지 로드
        input_image_path = os.path.join(input_image_folder, file)
        output_image_path = os.path.join(output_image_folder, file)
        img = cv2.imread(input_image_path)
        height, width, channels = img.shape

        # 사람 탐지
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # 탐지된 객체의 정보를 저장할 리스트
        class_ids = []
        confidences = []
        boxes = []

        # 탐지된 객체 분석
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if class_id == classes.index("person") and confidence > 0.5:
                    # 탐지된 객체의 바운딩 박스 좌표 계산
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-maximum suppression을 이용한 중복 박스 제거
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        match outfocus_no:
            case 0 :
                # 사람 영역을 마스크로 생성
                mask = np.zeros((height, width), dtype=np.uint8)

                for i in indices:
                    x, y, w, h = boxes[i]
                    mask[y:y+h, x:x+w] = 255

                # 마스크를 활용해 사람 영역을 제외한 영역에 블러 효과 적용
                blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
                output_img = np.where(mask[:, :, np.newaxis] == 255, img, blurred_img)

            case 1:
                    # 사람 영역을 따라 마스크 생성
                mask = np.zeros(img.shape[:2], np.uint8)
                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                for i in indices:
                    x, y, w, h = boxes[i]
                    rect = (x, y, w, h)
                    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

                # 확실한 배경과 확실한 전경 픽셀을 0 또는 1로 변경
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                img_fg = img * mask2[:, :, np.newaxis]

                    # 마스크를 활용해 사람 영역을 제외한 영역에 블러 효과 적용
                blurred_img = cv2.GaussianBlur(img, (21, 21), 0)
                output_img = np.where(mask2[:, :, np.newaxis] == 1, img, blurred_img)

        # 결과 이미지 저장
        cv2.imwrite(output_image_path, output_img)

def apply_af_train():
    # 1. 데이터 로드 및 라벨링
    data_dir = df_FileFolder.Item[2] # 각 폴더가 들어있는 상위 폴더 경로
    
    # 현재시간 가져와 결과 파일만들기
    now_day = datetime.today().strftime('%Y%m%d')
    model_save_path = now_day + '_cnn_model.h5'
    img_width, img_height = 150, 150
    batch_size = 32
    epochs = 50

    # 2. 데이터 생성기
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # 3. CNN 모델 구성
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 4. 모델 학습
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )

    # 5. 모델 저장
    model.save(model_save_path)
    lbPath[3].delete('1.0', tk.END)
    lbPath[3].insert(tk.INSERT, chars=model_save_path)
    update_Master(3,model_save_path)

def apply_af():
    # 1. 데이터 로드 및 라벨링
    model_save_path = df_FileFolder.Item[3]
    
    # 6. 새로운 이미지로 예측
    test_image_path = df_FileFolder.Item[4]
    predict_image(test_image_path, model_save_path)

def predict_image(image_path, model_path):
    data_dir = df_FileFolder.Item[2] # 각 폴더가 들어있는 상위 폴더 경로
    model = load_model(model_path)
    img_width, img_height = 150, 150
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    batch_size = 32
    
    predictions = model.predict(img_array)

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    class_indices = train_generator.class_indices
    class_labels = list(class_indices.keys())
    result_txt = 'Animal Test Result : '

    for label, probability in zip(class_labels, predictions[0]):
        result_txt = result_txt + '\n' + f"{label}: {probability:.4f}"
    
    tk.messagebox.showinfo(title = "Animal Test Result", message = result_txt)
    return predictions[0]


### GUI용 함수
# GUIMaster Data 업데이트 : 
def update_Master(idx, var):
    df_FileFolder.Item[idx] = var
    df_FileFolder.to_csv('GUIMaster.csv', index=False)

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
win.title('Python image editor')

# Frame 설정하기
frameF = tk.Frame(win, pady=5, width = 590, padx = 5)
frameF.grid(row=0, column=0, sticky= "ew", padx=5,pady=5)
frameNoise = tk.LabelFrame(win, text="Remove Noise", pady=5, width = 590, padx = 5)
frameNoise.grid(row=1, column=0, sticky= "ew", padx=5, pady=5)
frameEdge = tk.LabelFrame(win, text="Edge Detection", pady=5, width = 590, padx = 5)
frameEdge.grid(row=2, column=0, sticky= "ew", padx=5, pady=5)
frameOutFocus = tk.LabelFrame(win, text="Human reinforce", pady=5, width = 590, padx = 5)
frameOutFocus.grid(row=3, column=0, sticky= "ew", padx=5, pady=5)
frameAnimalFace = tk.LabelFrame(win, text="Animal Face", pady=5, width = 590, padx = 5)
frameAnimalFace.grid(row=4, column=0, sticky= "ew", padx=5, pady=5)

# 폴더 경로 설정 GUI
lbFame = []
lbPath = []
btnPath =[]

for i,x in enumerate(label_f):
    match i:
        case 0 | 1 :
            lbFame.append(tk.Label(frameF, text=x, width=15,padx =5, pady = 5))
            lbPath.append(tk.Text(frameF, width = 50, height = 1, padx =5, pady = 5, background='lightgrey'))
            btnPath.append(tk.Button(frameF, text="Change Path", width=10, padx =5, pady = 5, command=lambda i=i: onClick(i,fileyn[i])))
        case 2 | 3 | 4 :
            lbFame.append(tk.Label(frameAnimalFace, text=x, width=15, padx =5, pady = 5))
            lbPath.append(tk.Text(frameAnimalFace, width = 50, height = 1, padx =5, pady = 5, background='lightgrey'))
            btnPath.append(tk.Button(frameAnimalFace, text="Change Path", width=10, padx =5, pady = 5, command=lambda i=i: onClick(i,fileyn[i])))

    # 폴더/파일 이름 초기값 넣기
    lbPath[i].insert(tk.INSERT, chars=df_FileFolder.Item[i])

    lbFame[i].grid(row=i, column=0, padx =5, sticky=tk.W)
    lbPath[i].grid(row=i, column=1, padx =5, sticky=tk.W)
    btnPath[i].grid(row=i, column=2, padx =5, sticky=tk.W)

# Noise Frame 기능 구현
Filter_var = tk.IntVar()
Filter_var.set(0)
radio1 = tk.Radiobutton(frameNoise, text="Gaussian", variable=Filter_var, width = 17, padx =1, value=0)
radio2 = tk.Radiobutton(frameNoise, text="Median", variable=Filter_var,width = 17, padx = 1, value=1)
radio3 = tk.Radiobutton(frameNoise, text="Bilateral", variable=Filter_var,width = 17, padx = 3, value=2)
radio1.grid(row=0, column=0, sticky=tk.W)
radio2.grid(row=0, column=1, sticky=tk.W)
radio3.grid(row=0, column=2, sticky=tk.W)
radio1.select()
radio2.deselect()
button_filter = tk.Button(frameNoise, text="Apply Filter",width = 18, padx=5, command=lambda: apply_filter(Filter_var.get()))
button_filter.grid(row=0, column=3, sticky=tk.E)

# Edge Frame 기능 구현
Edge_var = tk.IntVar()
Edge_var.set(0)
radio4 = tk.Radiobutton(frameEdge, text="Canny", variable=Edge_var,width = 28, padx=1, value=0)
radio5 = tk.Radiobutton(frameEdge, text="Contour", variable=Edge_var, width = 28, padx = 1, value=1)
radio4.grid(row=0, column=0, sticky=tk.W)
radio5.grid(row=0, column=1, sticky=tk.W)
radio4.select()
radio5.deselect()
button_edge = tk.Button(frameEdge, text="Apply Edge", width = 18, padx=3, command=lambda: apply_edge(Edge_var.get()))
button_edge.grid(row=0, column=2, sticky=tk.E)

# OutFocus Frame 기능 구현
OF_var = tk.IntVar()
OF_var.set(0)
radio6 = tk.Radiobutton(frameOutFocus, text="Rectangle", variable=OF_var, width = 28, padx = 1, value=0)
radio7 = tk.Radiobutton(frameOutFocus, text="Boundary", variable=OF_var,width = 28, padx = 1, value=1)
radio6.grid(row=0, column=0, sticky=tk.W)
radio7.grid(row=0, column=1, sticky=tk.W)
radio6.select()
radio7.deselect()
button_of = tk.Button(frameOutFocus, text="Apply OutFocus", width = 18, padx = 3, command=lambda: apply_of(OF_var.get()))
button_of.grid(row=0, column=2, sticky=tk.E)

# Animal Face Frame 기능 구현
button_train = tk.Button(frameAnimalFace, text="Training", command=lambda: apply_af_train())
button_train.grid(row=5, column=0, sticky=tk.E, pady = 10) 
button_af = tk.Button(frameAnimalFace, text="Apply Animal Face", command=lambda: apply_af())
button_af.grid(row=5, column=1, sticky=tk.E)

win.mainloop()