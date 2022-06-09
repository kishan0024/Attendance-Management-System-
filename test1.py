# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
import csv

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import datetime

from keras.preprocessing import image

#loading data of students





# Loading the cascades
face_cascade=cv2.CascadeClassifier("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml")
trainimagelabel_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\training_data.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read(trainimagelabel_path)
except:
    e = "Model not found,please train model"


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces is ():
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]
        # gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
        # _,confidence=recognizer.predict(gray[y : y + h, x : x + w])
        # gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cropped_face


# Doing some Face Recognition with the webcam
def show_image():
    video_capture = cv2.VideoCapture(0)
    csv_file_data=[]
    while True:
        _, frame = video_capture.read()
        # canvas = detect(gray, frame)
        # image, face =face_detector(frame)

        face = face_extractor(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if type(face) is np.ndarray:

          for (x, y, w, h) in faces:
              global Id

              Id, conf = recognizer.predict(gray[y: y + h, x: x + w])
              # print(conf)
              temp=[]
              if conf < 70:
                  temp.append("dummy")
                  temp.append(Id)
                  temp.append("sub")
                  temp.append(str(datetime.datetime.now()))
                  # print(Id)
                  frame = cv2.putText(
                      img=frame,
                      text=str(Id),
                      org=(100,100),
                      fontFace=cv2.FONT_HERSHEY_DUPLEX,
                      fontScale=3,
                      color=(125, 246, 55),
                      thickness=2
                  )
                  csv_file_data.append(temp)
        cv2.imshow("img", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()


    #writing into CSV file
    fields=["name","id","subject","date","time"]
    filename="demo.csv"
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(csv_file_data)


show_image()

