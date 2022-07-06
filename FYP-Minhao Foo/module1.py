#--------------------------GUI-------------------------
from tkinter import *
import tkinter.messagebox as tsmg
#-------------------------Keylogger--------------------------
import pandas as pd
from datetime import datetime
from pynput.keyboard import Key, Listener
from pynput.mouse import Listener

#------------------------Mouselogger------------------------
import mouse
import math
import pyautogui as py
import time
import numpy as np
import threading

#-----------------------Naive bayes-------------------------------
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import datasets

#-----------------------------------------Fuzzy Logic------------------------------------
import skfuzzy as fuzz
from skfuzzy import control

#---------------------------------------Manhattan Distance---------------------------------------
from scipy.spatial.distance import cityblock

#---------------------------------------Euclidean Distance--------------------------------
from scipy.spatial.distance import euclidean
#-----------------------------Database--------------------------------
import pyodbc
#------------------------------------------------------------------




root = Tk()

df = pd.DataFrame(columns=['timeP','timeR','key','event','duration'])
start = time.time()
dfmouse = pd.DataFrame(columns=['x','y','time', 'Distance', 'velocity'])
clicked=threading.Event()

conn = pyodbc.connect('Driver={SQL SERVER};'
                      'Server=DESKTOP-6VAEH1I;'
                      'Database=userinfo;'
                      'Trusted_Connection=yes;')

#SERVER NAME HAS TO BE CHANGED WHEN USED BY ANOTHER USER

c=conn.cursor()
c.execute("""IF NOT EXISTS(SELECT * from sys.tables where name = 'usertest1')
CREATE TABLE usertest1 (username VARCHAR(50) PRIMARY KEY, passcode VARCHAR(50) NOT NULL, keyaccuracy FLOAT, mouseaccuracy FLOAT, manhattan FLOAT, euclidean FLOAT)""")
c.commit()

verivariable=[]

#Functions---------------------------------------------------------------

#--------------------------------Keylogger------------------------------------------
def on_press(key):
    global df
    key = key.char
    start = datetime.now() #Down-time
    timing = start.microsecond / 1000
    #dd = downdown(key)
    #result = dd.microsecond * 1000
    print(key)
    df = df.append({'timeP': timing, 'key':key , 'event': 'p'}, ignore_index=True)
    return start

def on_release(key):
    global df
    begin = on_press(key)
    test = datetime.now() #Up-time
    end = test.microsecond / 1000
    result = ((test-begin).total_seconds() * 1000) #Down-up Time

    df = df.append({'timeR':end, 'key': key.char, 'event': 'r', 'duration': result}, ignore_index=True)
    if key == Key.esc:
        return False
    return end


#------------------------------Verification-------------------------------------------------------------------
def check():
    c = conn.cursor()
    qry="SELECT * FROM usertest1 WHERE username=(?) AND passcode=(?)" 
    #Get Username and password
    username = uname.get()
    password=pwd.get()
    confirm = cpwd.get()
    if username=="" or password=="" or confirm == "":
        tsmg.showerror("Empty field", "Field cannot be empty")

    elif confirm != password:
        tsmg.showerror("Error", "Invalid username or password")

    else:
        c.execute(qry,(username,password))
        row=c.fetchone()

        if row==None:
            tsmg.showerror("Error", "Invalid username and password")

        else:
            bioVerify()


    uname.set("")
    pwd.set("")
    cpwd.set("")

def Register():
    c = conn.cursor()
    #Get Username and password
    username = uname.get()
    password=pwd.get()
    confirm = cpwd.get()
    if username=="" or password=="" or confirm == "":
        tsmg.showerror("Empty field", "Field cannot be empty")
    elif confirm != password:
        tsmg.showerror("Error", "Incorrect password or username")
    else:
        qry="SELECT * FROM usertest1 WHERE username=(?)"
        c.execute(qry, username)
        results=c.fetchone()
        if results!= None:
            tsmg.showerror("Error", "Username already exists")
        else:
            bioRegister()
        

    uname.set("")
    pwd.set("")
    cpwd.set("")


def savetocsv(daf):
    global uname
    un = uname.get()
    daf=df.to_csv(f'D:/source/repos/PythonApplication1/{un}.csv', index=False)

def saveveri(dbf):
    global uname
    un1 = uname.get()
    dbf=df.to_csv(f'D:/source/repos/PythonApplication1/verify.csv', index=False)

def saveMouseToCsvveri():  
    dfmouse.to_csv(r'D:\source\repos\PythonApplication1\mouseverify.csv', index = False)

#-------------------------------Keystroke dynamics--------------------------------------

def naiveBayes():
    global uname, pwd, accuracy,df
    uname1=uname.get()
    pswd=pwd.get()
    df = pd.read_csv(f'D:/source/repos/PythonApplication1/{uname1}.csv')
    target_names = np.array(['Positive', 'Negative'])

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75



    train = df[df['is_train']==True]
    test = df[df['is_train']==False]

    number = LabelEncoder()
    df['timeP'] = number.fit_transform(df['timeP'])
    df['timeR'] = number.fit_transform(df['timeR'])
    df['key'] = number.fit_transform(df['key'])
    df['event'] = number.fit_transform(df['event'])
    df['duration'] = number.fit_transform(df['duration'])


    features = ["timeP","timeR", "key", "event", "duration"]
    target = "duration"

    features_train, features_test, target_train, target_test = train_test_split(df[features],
    df[target], 
    test_size = 0.22,
    random_state = 54)

    model = GaussianNB()
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    accuracy = accuracy_score(target_test, pred)
    df = df.append({'accuracy':accuracy},ignore_index=True)
    with open("testNB.txt", "w") as f:
        f.write(f"{pred},{accuracy}")
    return accuracy


def naiveBayesveri():
    global uname, pwd, acc1
    uname1=uname.get()
    pswd=pwd.get()
    df = pd.read_csv('D:/source/repos/PythonApplication1/verify.csv')
    target_names = np.array(['Positive', 'Negative'])

    df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75



    train = df[df['is_train']==True]
    test = df[df['is_train']==False]

    number = LabelEncoder()
    df['timeP'] = number.fit_transform(df['timeP'])
    df['timeR'] = number.fit_transform(df['timeR'])
    df['key'] = number.fit_transform(df['key'])
    df['event'] = number.fit_transform(df['event'])
    df['duration'] = number.fit_transform(df['duration'])


    features = ["timeP","timeR", "key", "event", "duration"]
    target = "duration"

    features_train, features_test, target_train, target_test = train_test_split(df[features],
    df[target], 
    test_size = 0.22,
    random_state = 54)

    model = GaussianNB()
    model.fit(features_train, target_train)

    pred = model.predict(features_test)
    acc1 = accuracy_score(target_test, pred)
    df = df.append({'accuracy':acc1},ignore_index=True)
    return acc1

#-------------------------------Mouse Dynamics--------------------------
def mousedynamictest():
   
    with Listener(on_move = on_move,on_click = on_click ) as listener:
        listener.join()
    try:
        listener.wait()
        clicked.wait()
    finally:
        listener.stop()


def on_move(x, y):
    global dfmouse
    last = start
    vtime =time.time() - last
    print(x, y)
    distance = math.sqrt((y**2) + (x**2))
    print(distance, vtime)
    velo=distance/vtime
    last = time.time()    
    dfmouse=dfmouse.append({'x': x, 'y': y,'time':vtime, 'Distance': distance, 'velocity':velo}, ignore_index = True)

def on_click(x, y, button, pressed):
    print('{0} at {1}'.format(
        'Pressed' if pressed else 'Released',
        (x, y)))
    clicked.set()
    return False

def averagemousevalues():
    global dfmouse
    meandis=dfmouse['Distance'].mean()
    meanvel=dfmouse['velocity'].mean()
    dfmouse=dfmouse.append({'Avg Dist':meandis, 'Avg velo': meanvel}, ignore_index=True)
    print(meandis,meanvel)

def saveMouseToCsv():  
    dfmouse.to_csv(r'D:\source\repos\PythonApplication1\mousetest.csv', index = False)


def time_convert(sec):
    sec = sec % 60
    print("time lapsed = {0}".format(sec))



#---------------------------------------------------Manhattan Distance---------------------------------
def manhattanDist():
    global uname, pwd, x
    username = uname.get()
    def timepress():
        col=["timeP"]
        timePress = []
        df = pd.read_csv(f'D:/source/repos/PythonApplication1/{username}.csv',usecols=col)
        df = df.replace('',np.nan)
        df=df.dropna()
        for i in df.timeP:
                timePress.append(i)
        return timePress

    def timerelease():
        col=["timeR"]
        timeRelease = []
        df = pd.read_csv(f'D:/source/repos/PythonApplication1/{username}.csv',usecols=col)
        df = df.replace('',np.nan)
        df=df.dropna()
        for i in df.timeR:
                timeRelease.append(i)
        return timeRelease

    a = timepress()
    b = timerelease()
    dfMD = pd.DataFrame({'a':a,
                         'b':b})
    x=cityblock(dfMD.a,dfMD.b)
    return x

def manhattanDistVerify():
    global uname, pwd, y
    username = uname.get()
    def timepress():
        col=["timeP"]
        timePress = []
        timeRelease = []
        df = pd.read_csv(f'D:/source/repos/PythonApplication1/verify.csv',usecols=col)
        df = df.replace('',np.nan)
        df=df.dropna()
        for i in df.timeP:
                timePress.append(i)
        return timePress

    def timerelease():
        col=["timeR"]
        timeRelease = []
        timeRelease = []
        df = pd.read_csv(f'D:/source/repos/PythonApplication1/verify.csv',usecols=col)
        df = df.replace('',np.nan)
        df=df.dropna()
        for i in df.timeR:
                timeRelease.append(i)
        return timeRelease

    a = timepress()
    b = timerelease()
    dfMD = pd.DataFrame({'a':a,
                         'b':b})
    y=cityblock(dfMD.a,dfMD.b)
    return y

#-------------------------------------------Euclidean Distance------------------------------
def euclidDist():
    global uname, pwd, avgeuclid
    col=["x","y"]
    df=pd.read_csv("D:/source/repos/PythonApplication1/mousetest.csv",usecols=col)
    df = df.replace('',np.nan)
    df=df.dropna()
    euclid=[]
    loc1=0
    while loc1<=len(df)-2:
      dist=euclidean(df.iloc[loc1],df.iloc[loc1+1])
      euclid.append(dist)
      loc1+=1
    avgeuclid=sum(euclid)/len(euclid)
    return avgeuclid

def euclidDistVerify():
    global uname, pwd, average_euclid
    col=["x","y"]
    df=pd.read_csv('D:/source/repos/PythonApplication1/mouseverify.csv',usecols=col)
    df = df.replace('',np.nan)
    df=df.dropna()
    euclid=[]
    loc1=0
    while loc1<=len(df)-2:
      dist=euclidean(df.iloc[loc1],df.iloc[loc1+1])
      euclid.append(dist)
      loc1+=1
    average_euclid=sum(euclid)/len(euclid)
    return average_euclid

#--------------------------------------------------------Fuzzy Logic----------------------------------------------------------
def fuzzyLogic():
    nb = naiveBayes()
    md = manhattanDist()
    ed = euclidDist()
    global uname, pwd, accuracy,avgeuclid, x, mouseregistergui
    uname1=uname.get()
    pswd=pwd.get()

    acc=accuracy
    avgEuclid = avgeuclid
    mandist = x
    mreggui = mouseregistergui
    def closeRegister():
        mouseregistergui.destroy()

    distance = control.Antecedent(np.arange(0,1000,200),'distance')
    velocity= control.Antecedent(np.arange(0,1000,200),'velocity')
    result=control.Consequent(np.arange(0,7000,500),'result')

    distance.automf(3)
    velocity.automf(3)


    #threshold for classification [starting point, peak point, end point]
    result['poor']=fuzz.trimf(result.universe,[0,1500,2500])
    result['good']=fuzz.trimf(result.universe,[2300, 2500, 3000])


    rule1=control.Rule(distance['poor'] | velocity['poor'], result['poor'])
    rule2 = control.Rule(distance['good'] | velocity['good'], result['good'])

    result_ctrl=control.ControlSystem([rule1, rule2])

    resultant=control.ControlSystemSimulation(result_ctrl)

    col=["Avg Dist", "Avg velo"]
    df = pd.read_csv("D:/source/repos/PythonApplication1/mousetest.csv", usecols=col)
    resultant.input['distance']=df.iloc[-1,0]
    resultant.input['velocity']=df.iloc[-1,1]


    resultant.compute()
    mouseresult=resultant.output['result']
    c.execute('''INSERT INTO usertest1(username, passcode, keyaccuracy, mouseaccuracy,manhattan,euclidean) VALUES (?,?,?,?,?,?)''',(uname1,pswd,accuracy,mouseresult,mandist,avgEuclid))
    conn.commit()
    tsmg.showinfo(title="Done", message="Registration Complete")
    closeRegister()


def fuzzyVerify():
    nb = naiveBayesveri()
    md = manhattanDistVerify()
    ed = euclidDistVerify()

    global uname, pwd, acc1,average_euclid, y,mouseverigui
    uname1=uname.get()
    pswd=pwd.get()
    acc=acc1
    avgEuclid = average_euclid
    mandistance = y
    mvgui = mouseverigui

    def switchtopattern():
        mvgui.destroy()
        stats()

    distance = control.Antecedent(np.arange(0,1000,200),'distance')
    velocity= control.Antecedent(np.arange(0,1000,200),'velocity')
    result=control.Consequent(np.arange(0,7000,500),'result')

    distance.automf(3)
    velocity.automf(3)


    #threshold for classification [starting point, peak point, end point]
    result['poor']=fuzz.trimf(result.universe,[0,1500,2500])
    result['good']=fuzz.trimf(result.universe,[2300, 2500, 3000])


    rule1=control.Rule(distance['poor'] | velocity['poor'], result['poor'])
    rule2 = control.Rule(distance['good'] | velocity['good'], result['good'])

    result_ctrl=control.ControlSystem([rule1, rule2])

    resultant=control.ControlSystemSimulation(result_ctrl)

    col=["Avg Dist", "Avg velo"]
    df = pd.read_csv("D:/source/repos/PythonApplication1/mouseverify.csv", usecols=col)
    resultant.input['distance']=df.iloc[-1,0]
    resultant.input['velocity']=df.iloc[-1,1]

    resultant.compute()
    mouseresult=resultant.output['result']
    verivariable.append(uname1)
    verivariable.append(pswd)
    verivariable.append(acc)
    verivariable.append(mouseresult)
    verivariable.append(mandistance)
    verivariable.append(avgEuclid)


    c.execute("SELECT * FROM usertest1 WHERE username=(?)", uname1)
    record=c.fetchone()

    if((verivariable[2]/record[2]) * 100 >= 97) and ((verivariable[3] / record[3]) * 100 >= 100):
        if(verivariable[2]/record[2] *100 >100):
            percentage = ((verivariable[2]/record[2]) * 100) - 100
            if(percentage > 12.5): 
                print((verivariable[2]/record[2]*100),percentage,(verivariable[3]/record[3]*100))
                tsmg.showinfo("Success", f"Welcome, {uname1}",)
                switchtopattern()
            else:
                print((verivariable[2]/record[2]*100),percentage,(verivariable[3]/record[3]*100))
                tsmg.showinfo("Login failed", "Typing or mouse movement is different, try again")
        else:
            print((verivariable[2]/record[2]*100),(verivariable[3]/record[3]*100))
            tsmg.showinfo("Success", f"Welcome {uname1}",) 
            switchtopattern()

    else:
        print((verivariable[2]/record[2]*100),(verivariable[3]/record[3]*100))
        tsmg.showinfo("Login failed", "Typing or mouse movement is different, try again")



#------------------------------------------------------------GUI--------------------------------------------------------------
def mouseRegister():
    global mouseregistergui
    def switchback():
        mouseregistergui.destroy()
        bioRegister()
        

    mouseregistergui = Tk()
    mouseregistergui.title("Mouse Check")
    mouseregistergui.geometry("1000x1000")
    l=Label(mouseregistergui,text="Almost there! Press the following buttons, starting from Button 1!")
    l1=Label(mouseregistergui, text="Please Right-click Button 5")
    l.pack()
    btn1=Button(mouseregistergui,text="Button 1", command=mousedynamictest).place(x=100,y=410)
    btn2=Button(mouseregistergui, text="Button 2").place(x=350,y=870)
    btn3=Button(mouseregistergui, text="Button 3", command=averagemousevalues).place(x=350, y=200)
    btn4=Button(mouseregistergui, text="Button 4", command = saveMouseToCsv).place(x=345, y=423)
    btn5=Button(mouseregistergui,text="Button 5", command=fuzzyLogic).place(x=230, y=230)


def bioRegister():
    def switchreg():
        Gui1.destroy()
        mouseRegister()
    Gui1 = Tk()
    Gui1.title("Almost there")
    Gui1.geometry("2000x2000")
    veritext = """You are about to begin reading Italo Calvino’s new novel, If on a winter’s night a traveler. 
    Relax, Concentrate, Dispel every other thought. Let the world around you fade. Best to close the door; the TV is always on in the next room."""
    l_verify=Label(Gui1,text = veritext)
    l = Label(Gui1, text="Type in the following text")
    l_follow_up=Label(Gui1, text="Then, press the ESC key before continuing", font='bold')
    typebox = Text(Gui1, height = 20, width = 60)
    typebox.bind("<KeyRelease>", on_release)
    typebox.bind("<Escape>", savetocsv)
    btnNext = Button(Gui1, text="Confirm", command = switchreg)
    btnNext.pack()
    l.pack()
    l_verify.pack()
    l_follow_up.pack()
    typebox.pack()
    Gui1.mainloop()


def bioVerify():
    def switch():
        Gui.destroy()
        mouseVerify()
    Gui = Tk()
    Gui.title("Almost there")
    Gui.geometry("2000x2000")
    veritext = "H3110 W0rld, Progr4mm3d t0 W0rk 4nD not T0 F331, N0t ev3n 5ur3 ThaT th15 is Real, Hello World."
    l = Label(Gui,text="Type in the following text")
    l_verify = Label(Gui,text = veritext)
    l_follow_up=Label(Gui, text="Then, press the ESC key before continuing", font='bold')
    typebox = Text(Gui,height = 20, width = 60)
    typebox.bind("<KeyRelease>", on_release)
    typebox.bind("<Escape>", saveveri)
    btnNext = Button(Gui,text="Confirm",command = switch)
    btnNext.pack()
    l.pack()
    l_verify.pack()
    l_follow_up.pack()
    typebox.pack()
    Gui.mainloop()


def mouseVerify():
    global mouseverigui
    def switchback():
        mouseverigui.destroy()
        bioVerify()
        

    mouseverigui = Tk()
    mouseverigui.title("Your Login patterns")
    mouseverigui.geometry("1000x1000")

    l=Label(mouseverigui,text="Almost there! Press the following buttons, starting from Button 1!")
    l.pack()
    btnSwitch = Button(mouseverigui, text = "Back", command = switchback).pack()
    btn1=Button(mouseverigui,text="Button 1", command=mousedynamictest).place(x=100,y=410)
    btn2=Button(mouseverigui, text="Button 2").place(x=350,y=870)
    btn3=Button(mouseverigui, text="Button 3", command=averagemousevalues).place(x=350, y=200)
    btn4=Button(mouseverigui, text="Button 4", command = saveMouseToCsvveri).place(x=345, y=423)
    btn5=Button(mouseverigui,text="Button 5", command=fuzzyVerify).place(x=230, y=230)
    return mouseverigui


def stats():
    def closeall():
        tsmg.showinfo("Thank you","Logged Out successfully")
        statgui.destroy()
        root.destroy()

    global uname, verivariable,acc1,y,average_euclid
    manhattanDistVerify()
    euclidDistVerify()
    euDist = average_euclid
    manDist = y
    unamestat = uname.get()
    veristat = verivariable
    c.execute("SELECT * FROM usertest1 WHERE username=(?)", unamestat)
    record=c.fetchone()
    statgui = Tk()
    statgui.title("Your Login patterns")
    statgui.geometry("600x600")

    nbsimilarity = (veristat[2]/record[2])*100
    flsimilarity = (veristat[3]/record[3])*100
    mdsimilarity = (veristat[4]/record[4])*100
    edsimilarity = (veristat[5]/record[5])*100

    usernameLabel = Label(statgui,text="Username: ", font="SegoeUI 12 bold").place(x=0,y=0)
    username = Label(statgui, text=unamestat, font="SegoeUI 12 bold").place(x=170,y=0)

    statname = Label(statgui, text = "Algorithm used", font = "SegoeUI 10").place(x = 0, y = 50)
    statnumber = Label(statgui, text = "Value", font = "SegoeUI 10").place(x = 300, y = 50)

    labelKeyAcc = Label(statgui, text = "Naive Bayes: ").place(x=0,y=100)
    keyAcc = Label(statgui, text=veristat[2]).place(x=300,y=100)
    nblabel = Label(statgui, text = "Similarity(Naive Bayes): ").place(x=0,y=150)
    nbsimilar = Label(statgui,text = nbsimilarity).place(x=300, y=150)

    labelManDist = Label(statgui, text = "Manhattan Distance: ").place(x=0,y=200)
    manDistValue = Label(statgui,text=manDist).place(x=300,y=200)
    mdlabel = Label(statgui,text = "Similarity(Manhattan Distance): ").place(x=0,y=250)
    mdsimilar = Label(statgui,text = mdsimilarity).place(x=300,y=250)

    labelMouseAcc = Label(statgui, text = "Fuzzy Logic: ").place(x=0, y=300)
    mouseAcc = Label(statgui, text = veristat[3]).place(x=300, y=300)
    fllabel = Label(statgui, text = "Similarity (Fuzzy Logic): ").place(x=0,y=350)
    flsimilar = Label(statgui,text = flsimilarity).place(x=300,y=350)

    labelEuclid = Label(statgui, text="Euclidean Distance: ").place(x=0,y=400)
    euclidValue = Label(statgui, text=euDist).place(x=300,y=400)
    edlabel = Label(statgui,text="Similarity (Euclidean Distance): ").place(x=0,y=450)
    edsimilar = Label(statgui,text = edsimilarity).place(x=300,y=450)
    buttoncloseall = Button(statgui, text = "Logout", command=closeall).pack()

    statgui.mainloop()



uname=StringVar()
pwd=StringVar()
cpwd=StringVar()

root.geometry("700x500")
root.title("Welcome")
f=Frame(root)
Label(f, text="Login to Continue", font="SegoeUI 14 bold", pady=5).pack()
e1=Entry(f,textvariable=uname,font="SegoeUI 14 bold",borderwidth=5,relief=SUNKEN).pack(padx=5,pady=5)
Label(f,text="Password",font="SegoeUI 14 bold",pady=5).pack()
e2=Entry(f,textvariable=pwd,font="SegoeUI 14 bold",borderwidth=5,relief=SUNKEN, show="*").pack(padx=5,pady=5)
l1=Label(f,text="Confirm-Password",font="SegoeUI 14 bold",pady=5).pack()
e3=Entry(f,textvariable=cpwd,font="SegoeUI 14 bold",borderwidth=5,relief=SUNKEN, show="*").pack(padx=5,pady=5)
f.pack()

f=Frame(root)
b1=Button(f,text="Login",font="SegoeUI 10 bold", command=check)
b1.pack()

b2=Button(f, text="Register", font="SegoeUI 10 bold", command=Register)
b2.pack()


f.pack()

f=Frame(root)
Label(f,text="Don't Have An Account Then Sign-Up",font="SegoeUI 14 bold",pady=5).pack()
f.pack()
root.mainloop()
