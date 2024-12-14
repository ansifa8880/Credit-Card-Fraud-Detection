import tkinter as tk
import tkinter.messagebox
import pandas as p
from tkinter import *
import mysql.connector
import mysql.connector as mysql
#from tkinter import filedialog as fd 
import tkinter.filedialog
from tkinter import ttk
from datetime import date 
import datetime
import os
from dateutil import relativedelta
from PIL import ImageTk,Image  
import pandas as p
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from array import *

from pandas import DataFrame
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier






class NewWin():
    
   def __init__(self):
       self.win = tk.Tk()

       self.win.geometry("600x400+300+100");
       self.win.title(" Credit Card Fraud Prediction ")
       self.win.configure(bg="#912388")
       self.canvas = tk.Canvas(self.win, width = 600, height = 400)  
       self.canvas.place(x=0,y=0);

       l2 = tk.Label(self.win,text=" Credit Card Fraud Prediction ",width=40,relief="raised",bg="darkblue",fg="white",font=("cambria",16,"bold"))
       l2.place(x=30,y=30)
                           
       self.b2 = tk.Button(self.win,text=" Training ",width=30,bg="#157823",fg="white",relief="raised",font=("cambria",13,"bold"),command=self.callback)
       self.b2.place(x=120,y=150)

       
       self.win.mainloop()

   def callback(self):
       data=pd.read_csv("creditcard.csv")
       #dataset = pd.read_csv('Social_Network_Ads.csv')
       n=data.shape;
       n=n[0];
       
       X = data.iloc[:, :-1].values
       y = data.iloc[:, -1].values

    # Splitting the dataset into the Training set and Test set
       from sklearn.model_selection import train_test_split
       X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
       print(X_train)
       print(y_train)
       print(X_test)
       print(y_test)

       classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
       classifier.fit(X_train, y_train)
       y_pred = classifier.predict(X_test)
       cm = confusion_matrix(y_test, y_pred)
       print(cm)
       
       deacc=accuracy_score(y_test, y_pred)
       print("accuracy=",deacc)
             
       self.l4 = tk.Label(self.win,text="",width=40,relief="solid",bg="red",fg="white",font=("cambria",14,"bold"))
       self.l4.place(x=50,y=250)
       self.l4.configure(text="Accuracy Of KNN="+str(deacc*100))


#       self.l5 = tk.Label(self.win,text="",width=40,relief="solid",bg="red",fg="white",font=("cambria",14,"bold"))
 #      self.l5.place(x=50,y=300)

  #     self.l5.configure(text="Test Accuracy="+str(test_data_accuracy*100))
       
    #   input_data=(58,1,0,114,318,0,2,140,0,4.4,0,3,1)
    #   input_data_as_numpy_array=np.asarray(input_data)
    #   input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
    #   prediction=model.predict(input_data_reshaped)
    #   print(prediction)

     #  if(prediction[0]==0):
     #      print('the person does not have a heart disease')
     #  else:
     #      print('the person has heart disease')

   def loading(self):
           self.dataload()

   def dataload(self):
       app=Load();
       
           


class Test1():
   def __init__(self):
       self.root = tk.Tk()
       self.root.geometry("780x530+300+100");
       self.root.title(" Credit Card Fraud Prediction ")
       self.root.configure(bg="#912388")
       self.canvas = tk.Canvas(self.root, width = 800, height = 600)  
       self.canvas.place(x=0,y=0);

       l2 = tk.Label(self.root,text=" Credit Card Fraud Prediction  ",width=44,relief="raised",bg="darkblue",fg="white",font=("Monotype Corsiva",18,"bold"))
       l2.place(x=80,y=50) 

       self.img1 = ImageTk.PhotoImage(Image.open("air2.jpg"))  
       l1 = tk.Label(self.root, image=self.img1,width=800,relief="ridge",fg="#323223",font=("cambria",14,"bold"))
       l1.place(x=0,y=00)


       b1 = tk.Button(self.root,text=" Training & Testing ",width=25,bg="blue",fg="white",relief="raised",font=("Monotype Corsiva",16,"bold"),command=self.createNewWindow)
       b1.place(x=320,y=430)

       self.root.mainloop()

   def createNewWindow(self): 
       self.root.destroy()
       app=NewWin()
        

app=Test1()
