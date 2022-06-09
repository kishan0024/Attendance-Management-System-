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
CSVstore="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\student_record.csv"

ids=[]
names=[]


finalpath=trainimage_path
for j in os.listdir(finalpath):
    print(j)
    id = (j).split('_')
    names.append(id[0])
    id = id[1].split('.')
    # print(id[0])
    id = int(id[0])
    ids.append(id)

        # print(name)
        # print(id)

print(ids)
print(names)

dict={'name':names,'id':ids}
df=pd.DataFrame(dict)

df.to_csv(CSVstore,index=True)