import numpy as np
import cv2 
import os
import tensorflow as tf
from face_recognisation import smallmodel
from donotknow import newstudent
print("If student is registered then press 1")
print("If student is not registered then press 2 for making new folder")
number=int(input("Enter the number you want to proceed with : "))
if number==1:
    print("You have choosen option 1")
    print("So student is already registered")
    img_width=150
    img_height=150
    eye_cascade = cv2.CascadeClassifier('/home/manik/Documents/haarcascade_eye.xml')
    face_cascade = cv2.CascadeClassifier('/home/manik/Documents/haarcascade_frontalface_default.xml')
    vedio_capture = cv2.VideoCapture(0)
    while(True):
        ret , img=vedio_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0) ,2)
            roi_gray = img[y:y+h,x:x+w]
            img_item='my_image.png'
            path='/home/manik/Documents/Current_images'
            cv2.imwrite(os.path.join(path,img_item), roi_gray)
            print(x,y,w,h)
        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            prediction=smallmodel.build(img_height, img_width)
            print(prediction)
            break
    vedio_capture.release()
    cv2.destroyAllWindows()
elif number ==2:
    print("You have choosen the option 2")
    print("So you have to register the student ")
    newstudent=newstudent.buildagain() 
