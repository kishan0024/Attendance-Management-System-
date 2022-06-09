import csv
import os, cv2
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image

haarcasecade_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml"
trainimage_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\dataset\\train\\"
testimage_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\dataset\\train\\"
trainimagelabel_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\training_data.yml"


def get_items_and_labels(trainimage_path):
        faces=[]
        ids=[]
        finalpath=trainimage_path
        for j in os.listdir(finalpath):
            print(j)
            for k in os.listdir(finalpath+j+"\\"):
                # print(k)
                pilImage=Image.open(finalpath+j+"\\"+k).convert("L")
                # break
                img = np.array(pilImage, "uint8")
                id=(j).split('_')
                print(id)
                id=id[1]
                print(id)
                # print(id[0])
                id=int(id)
                print(id)
                ids.append(id)
                faces.append(img)
            # print(name)
            # print(id)

        return faces,ids

# name,id=get_items_and_labels(trainimage_path)










def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, id = get_items_and_labels(trainimage_path)
    # faces,s_id=get_images_and_labels(student_names,id)
    recognizer.train(faces, np.array(id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"

    print(res)

train_model()