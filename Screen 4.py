from tkinter import *
import csv

allcorrect = [False,False,False,False,False,False,False]


def check_pclass():
    pclass = entry1.get()
    if pclass.isalpha() == True or pclass == '' or pclass not in ['1','2','3']:
        entry1.configure(fg = 'red')        
    else:
        entry1.configure(fg = 'green')
        allcorrect[0] = True
def check_sex():
    sex = entry2.get()
    if sex.isalpha() == False  or sex == '' or sex not in['m', 'f']:
        entry2.configure(fg = 'red')
    else:
        entry2.configure(fg = 'green')
        allcorrect[1] = True
def check_age():
    age = entry3.get()
    if age.isalpha() == True  or age == '' or int(age) not in range(1,101):
        entry3.configure(fg = 'red')
    else:
        entry3.configure(fg = 'green')
        allcorrect[2] = True
def check_sibsp():
    sibsp = entry4.get()
    if sibsp.isalpha() == True  or sibsp == '' or int(sibsp) not in range(0,16):
        entry4.configure(fg = 'red')
    else:
        entry4.configure(fg = 'green')
        allcorrect[3] = True
def check_parch():
    parch = entry5.get()
    if parch.isalpha() == True or parch == '' or int(parch) not in range(0,16):
        entry5.configure(fg = 'red')
    else:
        entry5.configure(fg = 'green')
        allcorrect[4] = True
def check_fare():
    fare = entry6.get()
    if fare.isalpha() == True or fare == '' or int(fare) <= 0:
        entry6.configure(fg = 'red')
    else:
        entry6.configure(fg = 'green')
        allcorrect[5] = True
def check_embarked():
    embarked = entry7.get().lower()
    if embarked.isalpha() == False or embarked == '' or embarked not in ['c', 'q', 's']:
        entry7.configure(fg = 'red')
    else:
        entry7.configure(fg = 'green')
        allcorrect[6] = True

def check_all():
    check = True
    for item in allcorrect:
        if item == False:
            check = False
    if check == True:
        profile = (entry1.get(),entry2.get(),entry3.get(),entry4.get(),
              entry5.get(),entry6.get(),entry7.get())
        tocsv(profile)
    else:
        btn8.configure(fg = 'red')
    
def tocsv(profile):
    if profile[2] == 'm':
        sex_male = 1
        sex_female = 0
    else:
       sex_female = 1
       sex_male = 0
    if profile[6].lower() == 's':
      embarked_s = 1
      embarked_c = 0
      embarked_q = 0
    elif profile[6].lower() == 'c':
      embarked_c = 1
      embarked_s = 0
      embarked_q = 0
    else:
      embarked_q = 1
      embarked_c = 0
      embarked_s = 0
    updatedprofile = (profile[0], profile[2], profile[3], profile[4], profile[5], sex_female
                      ,sex_male, embarked_c, embarked_q, embarked_s)
    file = open('temp.csv', 'w+')
    with file:
        write = csv.writer(file)
        headers = ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S')
        write.writerow(headers)
        write.writerow(updatedprofile)


window = Tk()
window.geometry('1120x630')
window.title('Predicting Survival of External Profile')
entry1 = Entry(window)
entry2 = Entry(window)
entry3 = Entry(window)
entry4 = Entry(window)
entry5 = Entry(window)
entry6 = Entry(window)
entry7 = Entry(window)
mainlabel = Label(window, text = '''Welcome to Profile Creation.
    Here you will be able to create your own profile with attributes 
    to find the predicted probability of survival for the profile.''', font = 'Helvetica 13 bold')
label1 = Label(window, text = 'Enter the subject\'s ticket class - (1 = 1st, 2 = 2nd, 3 = 3rd): ')
label2 = Label(window, text = 'Enter the subject\'s sex - (m or f):')
label3 = Label(window, text = 'Enter the subject\'s age - (0-100):')
label4 = Label(window, text = 'Enter the subject\'s number of siblings and spouses abord - (0-15):')
label5 = Label(window, text = 'Enter the subject\'s number of parents and children abord - (0-15):')
label6 = Label(window, text = '''Enter the subject\'s passenger fare; in 1912,
    £100 would be worth over £11000 today
    (typical fares- for 1st class: £50+ , for 2nd class: £20-£50 , for 3rd class: £20 and below):''')
label7 = Label(window, text = '''Enter the subject\'s port of boarding :
    c = Cherbourg, q = Queenstown, s = Southampton: ''')
gap1 = Label(window, text = '')
gap2 = Label(window, text = '')
gap3= Label(window, text = '')
btn1 = Button(window,text = 'Check', command = check_pclass)
btn2 = Button(window,text = 'Check', command = check_sex)
btn3 = Button(window,text = 'Check', command = check_age)
btn4 = Button(window,text = 'Check', command = check_sibsp)
btn5 = Button(window,text = 'Check', command = check_parch)
btn6 = Button(window,text = 'Check', command = check_fare)
btn7 = Button(window,text = 'Check', command = check_embarked)
btn8 = Button(window,text = 'Predict...', command = check_all)
gap1.grid(row = 1, column = 2)
gap2.grid(row = 2, column = 2)
gap3.grid(row = 3, column = 2)
mainlabel.grid(row = 4, column = 2)
label1.grid(row = 5, column = 1)
entry1.grid(row = 5, column = 2)
btn1.grid(row = 5, column = 3)
label2.grid(row = 6, column = 1)
entry2.grid(row = 6, column = 2)
btn2.grid(row = 6, column = 3)
label3.grid(row = 7, column = 1)
entry3.grid(row = 7, column = 2)
btn3.grid(row = 7, column = 3)
label4.grid(row = 8, column = 1)
entry4.grid(row = 8, column = 2)
btn4.grid(row = 8, column = 3)
label5.grid(row = 9, column = 1)
entry5.grid(row = 9, column = 2)
btn5.grid(row = 9, column = 3)
label6.grid(row = 10, column = 1)
entry6.grid(row = 10, column = 2)
btn6.grid(row = 10, column = 3)
label7.grid(row = 11, column = 1)
entry7.grid(row = 11, column = 2)
btn7.grid(row = 11, column = 3)
btn8.grid(row = 12, column = 2)
window.mainloop()
