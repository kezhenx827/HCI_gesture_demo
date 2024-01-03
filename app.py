#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import os
import time
#import math eijroijgoiejrg
import math 


from collections import Counter
from collections import deque
#from gtts import gTTS
# from playsound import playsound
#import playsound 

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui

from utils import CvFpsCalc

# models
from model import KeyPointClassifier_R
from model import KeyPointClassifier_L
from model import PointHistoryClassifier
from model import MouseClassifier

#get_args()主要用來創建一個參數解析器，用於處理用戶從命令行輸入的參數
def get_args():
    #創建解析器
    parser = argparse.ArgumentParser()

    #定義參數
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)
    
    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    
    #解析使用者的命令輸入，將使用者提供的參數保存到args
    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    #創建一個參數解析器
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    #目前還不知道use_brect是用在哪裡的
    use_brect = True

    # Camera preparation ###############################################################
    # These settings ensure that the camera is configured with the specified width and height for capturing frames
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    # Use MediaPipe library to create a "Hands" object for hand tracking
    # This library provides pre-trained models for hand detection and tracking
    
    # imports the 'hands' module from the MediaPipe library
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        # determines whether the hand tracking should be optimized for static images (True) or video frames (False).
        static_image_mode=use_static_image_mode,
        
        # indicating that the model should track only one hand.
        max_num_hands=1,
        
        # defines the minimum confidence score required for a hand detection to be considered valid.
        min_detection_confidence=min_detection_confidence,
        
        # represents the minimum confidence score required for the hand tracking to be considered reliable
        min_tracking_confidence=min_tracking_confidence,
    )

    #KeyPointClassifier_R, KeyPointClassifier_L is defined in "keypoint_classifier.py"
    #Please go to "Keypoint_classifier.py" to get more detail
    #In brief, keypoint_classifier_R and keypoint_classifier_L is used to understand the gesture (靜態手勢分類器)
    keypoint_classifier_R = KeyPointClassifier_R(invalid_value=8, score_th=0.4)
    keypoint_classifier_L = KeyPointClassifier_L(invalid_value=8, score_th=0.4)
    
    #mouse_classifier is defined in "mouse_classifier.py"
    #MouseClassifier, is designed to classify mouse actions based on a set of hand landmark coordinates.(辨認哪種手勢是表示滑鼠的功能)
    mouse_classifier = MouseClassifier(invalid_value=2, score_th=0.4)
    
    #PointHistoryClassifier is defined in "point_history_classifier.py"
    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    # start to read and judge hand gesture, and save the result in keypoint_classifier_labels and point_history_classifier_labels
    with open(
            'model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    # CvFpsCals is defined in 'cvfpscalc.py'
    # To calculate FPS(每秒偵數)
    cvFpsCalc = CvFpsCalc(buffer_len=3)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)
    mouse_id_history = deque(maxlen=40)

    # 靜態手勢最常出現參數初始化
    keypoint_length = 5
    keypoint_R = deque(maxlen=keypoint_length)
    keypoint_L = deque(maxlen=keypoint_length)

    # result deque
    rest_length = 300
    rest_result = deque(maxlen=rest_length)
    speed_up_count = deque(maxlen=3)
    
    # ========= 使用者自訂姿勢、指令區 =========
    # time.sleep(0.5)
    # keepadd = False

    # ========= 按鍵前置作業 =========
    mode = 1 #0
    presstime = presstime_2 = presstime_3 = resttime = presstime_4 = time.time()

    detect_mode = 1 #0
    what_mode =  'keyboard'    # 'mouse'
    landmark_list = 0
    pyautogui.PAUSE = 0

    # ========= 滑鼠前置作業 =========
    wScr, hScr = pyautogui.size()
    frameR = 100
    smoothening = 7
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0
    mousespeed = 1.5
    clicktime = time.time()
    #關閉 滑鼠移至角落啟動保護措施
    pyautogui.FAILSAFE = False

    i = 0
    finger_gesture_id = 0

    # ========= 主程式運作 =========
    while True:
        left_id = right_id = -1
        #取得當前的FPS值
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) 
        # 如果按下ESC鍵時，則結束迴圈(結束主程式)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        # number, mode = select_mode(key, mode)
        mode = 2 
        # 不需要以上函式（但要找到number & mode ) 
        # number 不知道是what 

        # Camera capture
        # 進行水平翻轉以鏡像顯示 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #########################
        # 這部分是手勢檢測的實現部分
        
        # 將影像從BGR轉換為RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # 禁止修改影像數據的權限
        image.flags.writeable = False
        # 進行手勢檢測
        results = hands.process(image)
        # 恢復修改影像數據的權限
        image.flags.writeable = True
        
        
        
        ####rest_result#####################################
        if results.multi_hand_landmarks is None: #表示沒有偵測到手部
            rest_id = 0
            rest_result.append(rest_id)
        if results.multi_hand_landmarks is not None: #偵測值不為空的情況下
            rest_id = 1
            rest_result.append(rest_id) 
        #使用 Counter(rest_result).most_common() 統計 rest_result 列表中各元素的出現頻率，並按照頻率從高到低排序。
        most_common_rest_result = Counter(rest_result).most_common() 

        # old version for 10 sec to rest mode####################
        #print(most_common_rest_result[0])
        # if most_common_rest_result[0][0] == 0 and most_common_rest_result[0][1] == 300:
        #     if detect_mode != 0:
        #
        #         print('Mode has changed')
        #         detect_mode = 0
        #         what_mode = 'Rest'
        #         print(f'Current mode => {what_mode}')

        # new version for 10 sec to rest mode###################
        if time.time() - resttime > 100 : #檢查是否過了一定的時間(10秒)
            if detect_mode != 0:        #判斷當前的檢測模式是否不是休息模式(!=0)
                detect_mode = 0
                what_mode = 'Sleep'
                print(f'Current mode => {what_mode}')

        ####rest_result####
        
        
        
        
        #  ####################################################################
        # print(most_common_rest_result)
        # 到"動態手勢常出現的ID之前"的程式碼處理手部檢測結果，計算手的邊界框、計算地標點、轉換為相對座標/標準化座標，並將結果寫入數據集文件
        # 如果手部是左手，則將左手的靜態手勢ID（hand_sign_id_L）設置為預測結果；如果是右手，則將右手的靜態手勢ID（hand_sign_id_R）設置為預測結果。
        # 根據手的動作，將相應的手部地標添加到動態手勢歷史列表中（point_history）。最後，如果動態手勢歷史列表的長度為 history_length * 2，則使用動態手勢分類器（point_history_classifier）預測動態手勢的ID（finger_gesture_id）。
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):

                # Bounding box calculation
                # calc_bounding_rect 函式計算手的邊界框（Bounding Box），即能夠包圍整個手的矩形區域
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                # Landmark calculation
                #calc_landmark_list 函式計算手的地標點的座標列表，landmark_list 包含了手的各個特定地標的座標。
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                # print(landmark_list)

                # Conversion to relative coordinates / normalized coordinates
                # pre_process_landmark 函式將手的地標點座標轉換為相對座標或標準化座標，以便進行後續的處理。
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                # Write to the dataset file
                #logging_csv 函式將處理後的地標和動態手勢歷史列表寫入數據集文件，用於記錄手勢數據。
                #logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)
                logging_csv( mode, 1 , pre_processed_landmark_list, pre_processed_point_history_list)
                # 靜態手勢資料預測 
                # 使用靜態手勢分類器 keypoint_classifier_R 和 keypoint_classifier_L 預測右手和左手的靜態手勢ID。
                # 使用滑鼠手勢分類器 mouse_classifier 預測滑鼠手勢的ID。
                hand_sign_id_R = keypoint_classifier_R(pre_processed_landmark_list)
                hand_sign_id_L = keypoint_classifier_L(pre_processed_landmark_list)              
                mouse_id = mouse_classifier(pre_processed_landmark_list)
                # print(mouse_id)
                
                # 根據手的偏好性（左手或右手）來確定手勢的ID。
                if handedness.classification[0].label[0:] == 'Left':
                    left_id = hand_sign_id_L
                
                else:
                    right_id = hand_sign_id_R    
                
                # 手比one 觸發動態資料抓取
                if right_id == 1 or left_id ==1:
                    point_history.append(landmark_list[8]) 
                else:
                    point_history.append([0, 0]) 

                # 動態手勢資料預測
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                # print(finger_gesture_id) # 0 = stop, 1 = clockwise, 2 = counterclockwise, 3 = move,偵測出現的動態手勢

                #以下是雯####################################################################################################
                #
                # 動態手勢最常出現id #########################################
                # Calculates the gesture IDs in the latest detection
                # Counter 計算history中手勢出現的頻率
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()

                #滑鼠的deque
                mouse_id_history.append(mouse_id)
                most_common_ms_id = Counter(mouse_id_history).most_common()
                # print(f'finger_gesture_history = {finger_gesture_history}')
                # print(f'most_common_fg_id = {most_common_fg_id}')

                # 靜態手勢最常出現id #########################################
                hand_gesture_id = [right_id, left_id]
                keypoint_R.append(hand_gesture_id[0])
                keypoint_L.append(hand_gesture_id[1])
                # print(keypoint_R) # deque右手的靜態id
                # print(most_common_keypoint_id) # 右手靜態id最大
                #如果右手有靜態手勢，統計右手手勢頻率，取頻率最高的手勢
                #先檢查右手是因為手的偏好性，可自行設定
                if right_id != -1: 
                    most_common_keypoint_id = Counter(keypoint_R).most_common()
                #如果右手沒有手勢就統計左手的手勢頻率，取頻率最高的手勢
                else:  
                    most_common_keypoint_id = Counter(keypoint_L).most_common()

                
                # print(f'keypoint = {keypoint}')
                # print(f'most_common_keypoint_id = {most_common_keypoint_id}')

                ###############################################################

                # Drawing 把手、手的關節、手勢解釋......視覺化呈現給使用者看
                #debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                #debug_image = draw_landmarks(debug_image, landmark_list)
                #debug_image = draw_info_text(
                #    debug_image, # 目前的影像
                #    brect,       # 手部邊界框
                #    handedness,  # 手的偏好姓
                    keypoint_classifier_labels[most_common_keypoint_id[0][0]],#獲取靜態手勢的解釋文字
                    point_history_classifier_labels[most_common_fg_id[0][0]], #獲取動態手勢的解釋文字
                
                #ressttime用在休息模式的計時，當靜態手勢ID在規定時間內沒有變化，就會進入休息模式
                resttime = time.time()
        else:
            point_history.append([0, 0])

        #畫出手在過去幾幀照片中的運動軌跡，添加文字訊息以用於檢測
        #debug_image = draw_point_history(debug_image, point_history)
        #debug_image = draw_info(debug_image, fps, mode, number)

        # 偵測是否有手勢 #########################################

        if left_id + right_id > -2: #有偵測到手勢的話
            if time.time() - presstime > 1: #確保操作在1秒內只能執行一次
                # control keyboard
                if detect_mode == 1:
                    #presstime_2:上次按鍵操作時間
                    if time.time() - presstime_2 > 1:
                        # 靜態手勢控制
                        # most_common_keypoint_id==2, 按下按鍵K 
                        control_keyboard(most_common_keypoint_id, 2, 'K', keyboard_TF=True, print_TF=True) #暫停
                        control_keyboard(most_common_keypoint_id, 0, 'right', keyboard_TF=True, print_TF=True)
                        control_keyboard(most_common_keypoint_id, 7, 'left', keyboard_TF=True, print_TF=True)
                        control_keyboard(most_common_keypoint_id, 9, 'C', keyboard_TF=True, print_TF=True) # cc 字幕 
                        control_keyboard(most_common_keypoint_id, 5, 'up', keyboard_TF=True, print_TF=True) # 音量上
                        control_keyboard(most_common_keypoint_id, 6, 'down', keyboard_TF=True, print_TF=True) # 音量下
                        presstime_2 = time.time()

                    # right右鍵
                    #手勢ID為0且出現5次
                    if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
                        # print(i, time.time() - presstime_4)
                        #如果計數器 i 的值為 3 且距離上次按鍵操作超過 0.3 秒，則執行按下 "l" 鍵的操作，並將計數器 i 重置為 0
                        if i == 3 and time.time() - presstime_4 > 0.3:
                            pyautogui.press('l')
                            i = 0
                            presstime_4 = time.time()
                        #如果計數器 i 的值為 3 且距離上次按鍵操作超過 0.25 秒，則執行按下 "l" 鍵的操作
                        elif i == 3 and time.time() - presstime_4 > 0.25:
                            pyautogui.press('l')
                            presstime_4 = time.time()
                        #如果距離上次按鍵操作超過 1 秒，則執行按下 "l" 鍵的操作，並增加計數器 i
                        elif time.time() - presstime_4 > 1:
                            pyautogui.press('l')
                            i += 1
                            presstime_4 = time.time()

                    # left左鍵
                    if most_common_keypoint_id[0][0] == 7 and most_common_keypoint_id[0][1] == 5:
                        # print(i, time.time() - presstime_4)
                        if i == 3 and time.time() - presstime_4 > 0.3:
                            pyautogui.press('j')
                            i = 0
                            presstime_4 = time.time()
                        elif i == 3 and time.time() - presstime_4 > 0.25:
                            pyautogui.press('j')
                            presstime_4 = time.time()
                        elif time.time() - presstime_4 > 1:
                            pyautogui.press('j')
                            i += 1
                            presstime_4 = time.time()
        #          圖像          文字       文字起始座標    字體               字體縮放 文字顏色 文字粗細  抗鋸齒效果
        cv.putText(debug_image, what_mode, (400, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        # Screen reflection ###################################JL##########################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return

def control_keyboard(most_common_keypoint_id, select_right_id, command, keyboard_TF=True, print_TF=True, speed_up=False):
    if speed_up == False:
        # 如果最常見的右手靜態手勢ID為指定的ID（select_right_id）且出現次數為5次
        if most_common_keypoint_id[0][0] == select_right_id and most_common_keypoint_id[0][1] == 5:
            if keyboard_TF:
                pyautogui.press(command)
            if print_TF:
                print(command)
    #快轉
    if speed_up == True:
        # 如果最常見的右手靜態手勢ID為0（未檢測到特定手勢）且出現次數為5
        if most_common_keypoint_id[0][0] == 0 and most_common_keypoint_id[0][1] == 5:
            print(i, time.time() - presstime_4)
            # 如果 i 為 3 且距離上次按鍵操作超過 0.3 秒，則執行右方向鍵按下操作，重置 i 和更新按鍵時間
            if i == 3 and time.time() - presstime_4 > 0.3:
                pyautogui.press('right')
                i = 0
                presstime_4 = time.time()
            # 如果 i 為 3 且距離上次按鍵操作超過 0.25 秒，則執行右方向鍵按下操作，更新按鍵時間
            elif i == 3 and time.time() - presstime_4 > 0.25:
                pyautogui.press('right')
                presstime_4 = time.time()
            # 如果距離上次按鍵操作超過 1 秒，則執行右方向鍵按下操作，增加 i 並更新按鍵時間
            elif time.time() - presstime_4 > 1:
                pyautogui.press('right')
                i += 1
                presstime_4 = time.time()
        # if speed_up == True:
        #     i += 1
        #     if i > 3:
        #         pyautogui.press(command)
        #         pyautogui.press(command)
        #         pyautogui.press(command)





    # print(most_common_keypoint_id)
    # elif select_left_id == -1:
    #     if right_id == -1 and left_id == select_right_id:
    #         if keyboard_TF:
    #             pyautogui.press(command)
    #         if print_TF:
    #             print(command)
    # elif select_right_id == -1:
    #     if left_id == -1 and right_id == select_left_id:
    #         if keyboard_TF:
    #             pyautogui.press(command)
    #         if print_TF:
    #             print(command)

#沒用到
def pick_gesture_command():
    left_number = input('left gesture number :')
    right_number = input('right gesture number :')
    command = input('what command :')
    return int(left_number), int(right_number), command

#沒用到
def pick_number(inputstring):
    keepask = True
    while keepask:
        try:
            number = input(f'{inputstring} :')
            number = int(number)
            if number < -1 or number > 3 or number == 0:
                raise Exception('number is not in range')
        except:
            print('choose again')

        else:
            keepask = False
            # print('choosing nicely')
    return number

def findDistance(p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
        cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
        cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]


if __name__ == '__main__':
    main()
