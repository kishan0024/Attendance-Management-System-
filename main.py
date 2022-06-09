import cv2
import os
import numpy as np
import pandas as pd




face_classifier=cv2.CascadeClassifier("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml")


def face_rec(img):
    faces=face_classifier.detectMultiScale(img,1.3,5)

    if faces==():
       return None

    for(x,y,w,h) in faces:
        x-=10
        y-=10
        cropped_img=img[y:y+h+50,x:x+w+50]

    return cropped_img


cap=cv2.VideoCapture(0)
count=0


username=input("enter name:")
id=input("enter id:")
os.mkdir('./dataset/train/'+username+'_'+id)
while True:

    ret,img=cap.read()

    if face_rec(img) is not None:
        count+=1
        face=cv2.resize(face_rec(img),(400,400))

        file_path='./dataset/train/'+username+'_'+id+'/'+username+'_'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("face cropp",face)
    else:
        print("face not found")
        pass
    if cv2.waitKey(1)==13 or count==100:
        break
cap.release()
cv2.destroyAllWindows()
print("doneeeee")
