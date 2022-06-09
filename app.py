from flask import Flask,render_template,Response,request, redirect, url_for, session
from flask_session import Session
from flask_mysqldb import MySQL
import mysql.connector
import re
import cv2
import numpy as np
import csv
import os
import numpy as np
import pandas as pd
import datetime
import time
from PIL import ImageTk, Image



app=Flask(__name__)
app.secret_key=os.urandom(24)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)



#paths for training label and data


haarcasecade_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml"
trainimage_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\dataset\\train\\"
testimage_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\dataset\\train\\"
trainimagelabel_path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\training_data.yml"



#path module ends



#sql connection
con=mysql.connector.connect(host="localhost",user="root",password="",database="de_pro")
# print(con)
cursor=con.cursor()
#sql connection ends






# Loading the cascades
face_cascade=cv2.CascadeClassifier("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\haarcascade_frontalface_default.xml")
trainimagelabel_path= "training_data.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    recognizer.read(trainimagelabel_path)
except:
    e = "Model not found,please train model"



#function to extract features from image
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



#function to be used in embadding it to website
    # global csv_file_data=[]
def show_image(start,end,sub):
    video_capture = cv2.VideoCapture(0)
    print(start)
    print(end)
    print(sub)
    csv_file_data=[]
    ids = []

    while True:
        if (start == -1 or end == -1 or start == 0):

            img = Image.open("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\imgs\\before.png")
            img = np.array(img, "uint8")
            ret, buffer = cv2.imencode('.jpg', img)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # print("askdjkjahsdka"
            return

        if(end==1 and start==2):

            print("c3")
            csv_file_data=[]

            img = Image.open("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\imgs\\before.png")
            img = np.array(img, "uint8")
            ret, buffer = cv2.imencode('.jpg', img)

            frame = buffer.tobytes()
            start=0
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            # print("askdjkjahsdka"
            return




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
              id=int(Id)
              temp = []
              if conf < 70:
                  cursor.execute("select s_name from student_info where s_id={}".format(id))

                  name=cursor.fetchall()
                  if len(name)>0:


                    nm=name[0][0]
                    temp.append(nm)
                    temp.append(id)
                    temp.append(sub)
                    temp.append(str(datetime.datetime.now()))
                    # cursor.execute("INSERT INTO `attendance`( `at_id`, `at_name`, `at_sib`) VALUES (?,'?','?')".format(int(id), str(name[0][0]), sub))

                  frame = cv2.putText(
                      img=frame,
                      text=str(id),
                      org=(100,100),
                      fontFace=cv2.FONT_HERSHEY_DUPLEX,
                      fontScale=3,
                      color=(125, 246, 55),
                      thickness=2
                  )
              fields = ["name", "id", "subject", "date", "time"]
              today = datetime.date.today()
              filename = 'C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\attendance\\'+sub + '_' + str(today)
              # print(filename)
              # print(temp)

              csv_file_data.append(temp)

              csv_file_data.append(temp)
              with open(filename, 'w') as csvfile:
                  # creating a csv writer object
                  csvwriter = csv.writer(csvfile)
                  # print("c2")
                  # writing the fields
                  csvwriter.writerow(fields)

                  # writing the data rows
                  csvwriter.writerows(csv_file_data)








        ret,buffer=cv2.imencode('.jpg',frame)
        frame=buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    # video_capture.release()
    # cv2.destroyAllWindows()

#ends

#function to get label and image arrays
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
                id=id[1].split('.')
                # print(id[0])
                id=int(id[0])
                ids.append(id)
                faces.append(img)
            # print(name)
            # print(id)

        return faces,ids

#function to get label and image arrays ends




#training the model

def train_model():
    session['train_start']="start_model"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(haarcasecade_path)
    faces, id = get_items_and_labels(trainimage_path)
    # faces,s_id=get_images_and_labels(student_names,id)
    recognizer.train(faces, np.array(id))
    recognizer.save(trainimagelabel_path)
    res = "Image Trained successfully"
    return render_template("admin_inter.html",train="done")
    print("chelc")

#training the model ends




@app.route('/')
def home():
    # if 'admin_id' in session:
    #     return render_template("admin.html")
    # elif 'student_id' in session:
    #     return render_template("student.html")
    # elif 'faculty_id' in session:
    #     return render_template("faculty.html")
    # else:
        return render_template("index.html")



# #capture_image funtion
# def capture_img():
#     return "hello"    


@app.route('/admin_final')
def admin_final():
    return render_template('admin.html')

@app.route('/video')
def video():
    sub="notsel"
    if not 'start_attendance' in session or not 'end_attendance' in session:
        start=-1
        end=-1

    if 'start_attendance' in session and 'end_attendance':
        start=int(session['start_attendance'])
        end=int(session['end_attendance'])
    if 'sub' in session:
        sub=str(session['sub'])

    return Response(show_image(start,end,sub),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/admin')
def admin():
   if 'admin_id' in session:
       return render_template('admin_inter.html')
   else:
       redirect('/')

@app.route('/faculty')
def faculty():
   if 'faculty_id' in session:
       return render_template('faculty.html')
   else:
       redirect('/')

@app.route('/student')
def student():
   if 'student_id' in session:
       return render_template('student.html')
   else:
       redirect('/')


@app.route('/login',methods=['GET', 'POST'])
def login():
    roles=request.form.get('roles')
    mail=request.form.get("Username")
    passw=request.form.get("password")
    print(roles)
    print(mail)
    print(passw)
    if roles=="admin":
        cursor.execute("SELECT * FROM `admin` WHERE `admin_email`='{}' and `admin_pass`='{}'".format(mail,passw))
        users=cursor.fetchall()

        if len(users)>0:
            session['admin_id']=users[0][1]
            return redirect('/admin')
        else:
            return render_template("index.html")
    elif roles=="faculty":
        cursor.execute("SELECT * FROM `faculty` WHERE `fac_email`='{}' and `fac_pass`='{}'".format(mail,passw))
        users = cursor.fetchall()
        print(users,"sd")

        if len(users) > 0:
            session['faculty_id']=users[0][0]
            return redirect('/faculty')
        else:
            return render_template("index.html")
    elif roles=="student":
        cursor.execute("SELECT * FROM `student_info` WHERE `s_email`='{}' and `s_password`='{}'".format(mail, passw))
        users = cursor.fetchall()
        if len(users) > 0:
            session['student_id']=users[0][0]
            return redirect('student')
        else:
            return render_template("index.html")





@app.route('/train_model_fun')
def train_model_fun():
    return train_model()


#registration function start







@app.route('/new_registration',methods=['GET', 'POST'])
def new_registration():
    id = request.form.get("s_id")
    name = request.form.get("s_name")
    enrollment = request.form.get("s_enroll")
    dob = request.form.get('s_dob')
    gender = request.form.get("s_gender")
    branch = request.form.get("s_branch")
    passwd = request.form.get("s_pass")
    contact = request.form.get("s_contact")
    conf_pass = request.form.get("conf_pass")
    email = request.form.get("s_email")
    p_name = request.form.get("p_name")
    p_contact = request.form.get("p_contact")
    p_email = request.form.get("p_email")

    print(dob)
    cursor.execute("INSERT INTO `student_info`VALUES ({},'{}','{}',{},{},'{}','{}','{}','{}')".format(id,name,email,contact,enrollment,dob,branch,gender,passwd))
    cursor.execute("INSERT INTO `parents_details`(`st_id`, `p_name`, `p_mobile`, `p_email`) VALUES ({},'{}',{},'{}')".format(id,p_name,p_contact,p_email))
    con.commit()
    session['student_name']=name
    session['student_id']=id
    return render_template('admin.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')



#face capture starts

def face_rec(img):
    faces=face_cascade.detectMultiScale(img,1.3,5)

    if faces==():
       return img

    for(x,y,w,h) in faces:
        x-=10
        y-=10
        cropped_img=img[y:y+h+50,x:x+w+50]

    return cropped_img


def data_cap_util(username,id):




        cap = cv2.VideoCapture(0)
        count = 0



        print("check")
        os.mkdir('./dataset/train/' + username + '_' + id)
        while True:

            ret, img = cap.read()

            if face_rec(img) is not None:
                count += 1
                face = cv2.resize(face_rec(img), (400, 400))

                file_path = './dataset/train/' + username + '_' + id + '/' + username + '_' + str(count) + '.jpg'
                cv2.imwrite(file_path, face)

                cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                # cv2.imshow("face cropp", face)
            else:
                # print("face not found")
                pass
            if count == 100:
                # session['cap_done']='true'
                img=Image.open("C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\imgs\\theend.png")
                img = np.array(img, "uint8")
                ret, buffer = cv2.imencode('.jpg', img)

                frame = buffer.tobytes()
                yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                print("askdjkjahsdka")
                return

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/data_capture')
def data_capture():
    if 'student_name' in session:
        username = session['student_name']
        id = session['student_id']
    return Response(data_cap_util(username,id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logout')
def logout():
    session.pop('admin_id',None)
    session.pop('student_id', None)
    session.pop('student_name', None)
    return render_template('index.html')


@app.route('/start_fun',methods=['GET', 'POST'])
def start_fun():
    sub=request.form.get("sub")
    print(sub)
    session['sub']=sub
    session['start_attendance']=1
    session['end_attendance'] = 0

    return redirect('/faculty')

@app.route('/end_fun')
def end_fun():
    session['end_attendance']=1
    session['start_attendance'] = 2
    return redirect('/faculty')

#
@app.route('/upload_csv')
def upload_csv():
    path="C:\\Users\\mahet\\PycharmProjects\\DE_2nd\\attendance\\"

    for j in os.listdir(path):
        print(j)


    return render_template('admin_inter.html')
if __name__=="__main__":
    app.run(debug=True)