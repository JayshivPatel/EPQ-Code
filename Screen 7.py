from tkinter import *
import csv

window = Tk()
window.geometry('480x270')
window.title('Train Data')
frame = Frame(window)

scrollbar = Scrollbar(frame, orient = VERTICAL)
listbox = Listbox(frame, width = 70, yscrollcommand = scrollbar.set)

count = 1
file = open('train.csv', 'r')
for line in file:
    listbox.insert(count, line)
    count+=1


scrollbar.pack(side=RIGHT, fill=Y)
btn1 = Button(window, text = 'Ok', command = None)
scrollbar.config(command = listbox.yview)
scrollbar.pack(side=RIGHT, fill = Y)
listbox.pack()
frame.pack(padx = 30, pady = 30)
btn1.pack(side= BOTTOM)
window.mainloop()
