from tkinter import *
import tkinter as tk
#Create an instance of Tkinter Frame
win = Tk()

name_var=tk.StringVar()
passw_var=tk.StringVar()
#Set the geometry
win.geometry("700x350")

#Set the default color of the window
win.config(bg='#4fe3a5')

menu= StringVar()
menu.set("Select Your Role")
drop= OptionMenu(win, menu,"Administrator","Faculty","Student").place(relx=.5, rely=.2,anchor= CENTER)
Label(win, text = "Student Attendence Management System", font= ('Helvetica 15 bold')).place(relx=.5, rely=.1,anchor= CENTER)
Label(win, text = "Enter Name:", font= ('Helvetica 13 bold')).place(relx=.3, rely=.3,anchor= CENTER)
name_entry= Entry(win,textvariable=name_var,font= ('Helvetica 13 bold')).place(relx=.6, rely=.3 ,anchor= CENTER)
Label(win, text = "Enter Password:", font= ('Helvetica 13 bold')).place(relx=.3, rely=.4,anchor= CENTER)
name_entry= Entry(win,textvariable=passw_var,font= ('Helvetica 13 bold')).place(relx=.6, rely=.4 ,anchor= CENTER)
Button
win.mainloop()