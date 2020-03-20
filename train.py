#  圖片辨識訓練資料  #

import sys # 系統
import numpy as np # 產生陣列
import cv2 # 做圖片處理
import imutils # 做圖片處理



#########################  圖片辨識前置作業  ########################

im = cv2.imread('train.png') # 讀取用來訓練圖片
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) # 將訓練圖片灰化
blur = cv2.GaussianBlur(gray,(5,5),0) # 將訓練圖片做高斯模糊，去除雜訊
cv2.imwrite("blur.png",blur) # 儲存圖片
thresh = cv2.adaptiveThreshold(blur,255,1,1,9,2) # 將訓練圖片做自適二極化，圖片將變成黑白
cv2.imwrite("tresh.png",thresh) # 儲存圖片



#########################  辨識圖片輪廓  ############################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # 找輪廓

samples =  np.empty((0,100)) # 產生空陣列
responses = [] # 用來儲存使用者按下的按鍵
keys = [i for i in range(48,58)] # 鍵盤按鍵

for cnt in contours:
    if cv2.contourArea(cnt) > 17: # 判斷輪廓面積
        [x,y,w,h] = cv2.boundingRect(cnt) # 將輪廓用矩形包起來，x,y是左上點的座標，w,h是寬和高
        
        if  h>6 and w <10 : # 判斷矩形高度
            
            cv2.rectangle(im,(x-1,y),(x+w,y+h),(0,0,255),1) # 畫出矩形
            num = thresh[y:y+h,x:x+w] # 抓取每個數字所在的位置
            cv2.imwrite("num.png",num) # 儲存
            renum = cv2.resize(num,(10,10)) # 將數字的圖片重設大小為10x10
            cv2.imshow("train",imutils.resize(im, width=500)) # 顯示訓練的圖片
            key = cv2.waitKey(0) # 防止沒有回應
            
            if key == 27:  # (ESC)
                cv2.destroyAllWindows() # 關掉視窗
                sys.exit() # 關閉程式
                
            elif key in keys:
                responses.append(int(chr(key))) # 儲存使用者按下的數字
                sample = renum.reshape((1,100)) # 將10x10數字圖片儲存在一個有100個值的陣列中
                samples = np.append(samples,sample,0) # 將所有數字圖片儲存下來
                
                
                
###########################  儲存要用來訓練的資料  ############################

responses = np.array(responses,np.float32) # 將使用者按下的數字轉成用array儲存
responses = responses.reshape((responses.size,1)) # 將使用者按下的數字個別用一個陣列儲存
np.savetxt('generalsamples.txt',samples) # 將數字的圖片儲存成txt
np.savetxt('generalresponses.txt',responses) # 將使用者按下的數字儲存成txt

print ("訓練完成")

cv2.destroyAllWindows() # 關掉視窗
sys.exit() # 關閉程式

