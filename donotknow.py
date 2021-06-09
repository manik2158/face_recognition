import numpy as np
import pandas as pd 
import cv2 
import os
class newstudent:
    def buildagain():
        cnt=0
        name=input("Enter the student name")
        path='/home/manik/Documents/images/'+name
        os.mkdir(path)
        eye_cascade = cv2.CascadeClassifier('/home/manik/Documents/haarcascade_eye.xml')
        face_cascade = cv2.CascadeClassifier('/home/manik/Documents/haarcascade_frontalface_default.xml')
        vedio_capture = cv2.VideoCapture(0)
        while(True):
            ret , img=vedio_capture.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img, 1.3, 5)
            for (x,y,w,h) in faces:
                eyesn = 0
                imgCrop = img[y:y+h,x:x+w]
                #cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                    eyesn = eyesn +1
                if eyesn >= 2:
                    #### increase the counter and save 
                    cnt +=1
                    img_item=name+'.'+str(cnt)+'.png'
                    cv2.imwrite(os.path.join(path,img_item), roi_gray)
                    print("Image captured ",end=" ")
                    print(cnt)
                    #cv2.imshow('img',imgCrop)
            cv2.imshow('img',img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        #cap.release() 
        print("All images have been processed!!!")
        vedio_capture.release()
        cv2.destroyAllWindows()
