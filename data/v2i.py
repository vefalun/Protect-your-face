import cv2
import os
import glob

MAE_path = sorted(glob.glob(os.path.join('./data/no_on_face_video', '*.mp4')))
print(MAE_path)
c = 0
for i in range(len(MAE_path)):
    filename = MAE_path[i]
    print(filename)
    vc = cv2.VideoCapture(filename) 
    rval=vc.isOpened()
    while rval:  
        c = c + 1
        rval, frame = vc.read()
        if rval:
            cv2.imwrite('./data/img/no_on_face_img/no_on_face_'+str(c) + '.jpg', frame) #存储为图像
        else:
            break
    vc.release()
        

MAE_path = sorted(glob.glob(os.path.join('./data/on_face_video', '*.mp4')))
print(MAE_path)
c = 0
for i in range(len(MAE_path)):
    filename = MAE_path[i]
    print(filename)
    vc = cv2.VideoCapture(filename) 
    rval=vc.isOpened()
    while rval:  
        c = c + 1
        rval, frame = vc.read()
        if rval:
            cv2.imwrite('./data/img/on_face_img/on_face_'+str(c) + '.jpg', frame) #存储为图像
            cv2.waitKey(1)
        else:
            break
    vc.release()