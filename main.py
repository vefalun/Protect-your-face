'''
Auther: Jialun Cai
2022/09/06
'''
import cv2
import datetime
import schedule
import glob

from locale import normalize
import os
from pickletools import optimize
from tkinter import Image

import torch
from torchvision import transforms as T
import pyautogui

from resnet import ResNet
import cv2
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'



def capture():
    print('开始运行')
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW) 
    # 获取当前时间
    now_time = datetime.datetime.now()
    timeStr = datetime.datetime.strftime(now_time,'%Y_%m_%d_%H_%M_%S')
    reg, frame = cap.read()    
    filename = timeStr + '.jpg' # filename为图像名字
    print(filename)
    
    cv2.imwrite('X:/face/img/'+filename, frame)
    cap.release()

    num = 0
    #算法利用前后三帧进行判断
    MAE_path = sorted(glob.glob(os.path.join(r'X:\face\img', '*.jpg')))
    while len(MAE_path)>3:
        dir = MAE_path[0]
        os.remove(dir)
        MAE_path.pop(0)
    model.eval()

    for i in range(3):
        dir = MAE_path[i]
        image = Image.open(dir)
        img = T.Resize((224,224))(image)

        
        img_input = T.ToTensor()(img).to(device)
        img_input = T.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])(img_input)
        img_input = img_input.unsqueeze(0)
        output = model(img_input)
        _, pred = torch.max(output, axis=1)
        print(pred)

        if pred == 1:
            num += 1
        if num > 1:
            pyautogui.alert("不要把手放在脸上")
            num = -3
            # print("手在脸上")



#载入模型
model = ResNet().to(device)
model.load_state_dict(torch.load("./save_model/best_model.pth",map_location=torch.device(device)))

schedule.every(1).second.do(capture)   # 每隔1秒执行一次
while True:
    schedule.run_pending()


    
