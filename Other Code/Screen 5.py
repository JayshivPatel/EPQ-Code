from tkinter import *
import csv

def check_answer():
    answer = entry1.get()
    if answer.isalpha() == True or answer == '' or int(answer) not in range(1,419):
        entry1.configure(fg = 'red')        
    else:
        entry1.configure(fg = 'green')
        fromcsv()

def fromcsv():
    answer = int(entry1.get())
    index = answer-1
    answer = answer+1
    file = open('test.csv', 'r')
    with file:
        read = csv.reader(file)
        for i in range(1, answer):
            next(read)
        row = next(read)
    var1 = 'You have chosen: ', row[2]
    var2 = 'Class: ', row[1]
    var3 = 'Sex: ',row[3]
    var4 = 'Age: ', row[4]
    var5 = 'Siblings/Spouses: ',row[5]
    var6 = 'Parents/Children: ', row[6]
    var7 = 'Fare: ', row[8]
    var8 = 'Embarked', row[10]
    label3 = Label(window, text = str(var1))
    label4 = Label(window, text = str(var2))
    label5 = Label(window, text = str(var3))
    label6 = Label(window, text = str(var4))
    label7 = Label(window, text = str(var5))
    label8 = Label(window, text = str(var6))
    label9 = Label(window, text = str(var7))
    label10 = Label(window, text = str(var8))
    label2.grid(row = 4, column = 1)
    label3.grid(row = 5, column = 1, pady = 10)
    label4.grid(row = 6, column = 1, pady = 10)
    label5.grid(row = 7, column = 1, pady = 10)
    label6.grid(row = 8, column = 1, pady = 10)
    label7.grid(row = 6, column = 2, pady = 10)
    label8.grid(row = 7, column = 2, pady = 10)
    label9.grid(row = 8, column = 2, pady = 10)

    
    print('Predicted probability of survival: NEED TO FILL IN THIS')
    

window = Tk()
window.geometry('640x360')
window.title('Predicting survival of internal profile')
label1 = Label(window, text = 'There are 418 profiles to select from. \n Please enter a number between 1 and 418:')
entry1 = Entry(window)
btn1 = Button(window, text = 'Predict', command = check_answer)
label2 = Label(window, text = 'You have chosen:', font = 'Helvetica 13 bold')

gap1 = Label(window, text = '')
gap2 = Label(window, text = '')
gap3= Label(window, text = '')
label1.grid(row = 1, column = 1, padx = 20)
entry1.grid(row = 1, column = 2, padx = 20)
btn1.grid(row = 1, column = 3)
gap1.grid(row = 2)
gap2.grid(row = 3)
gap3.grid(row = 4)

window.mainloop()
