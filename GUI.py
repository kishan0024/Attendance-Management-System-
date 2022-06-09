# Importing the libraries
from PIL import Image
from keras.applications.vgg16 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk,Image

#setting the path for diff usages
# Loading the cascades
face_cascade=cv2.CascadeClassifier("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml")
trainimagelabel_path= "/training_data.yml"
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
    return  cropped_face




#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("Digital Microscope")
window.config(background="#FFFFFF")

#Graphics window
imageFrame = tk.Frame(window, width=600, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)
def show_frame():
    _, frame = cap.read()
    g_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face=face_extractor(frame)
    # face_new=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(g_frame, 1.3, 5)
    if type(face) is np.ndarray:

      for (x, y, w, h) in faces:
          global Id
        #fetching result from model
          Id, conf = recognizer.predict(g_frame[y: y + h, x: x + w])
        #displaying id on screen

          if conf < 70:

              frame = cv2.putText(
                  img=frame,
                  text=str(Id),
                  org=(70, 70),
                  fontFace=cv2.FONT_HERSHEY_DUPLEX,
                  fontScale=2,
                  color=(125, 246, 55),
                  thickness=2
              )




    # frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)



#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 600, column=0, padx=10, pady=2)


show_frame()  #Display 2
window.mainloop()  #Starts GUI
