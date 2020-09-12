from tkinter import *
import tkinter.messagebox as box

window = Tk()
window.geometry('640x360')
window.title('Titanic Survival Menu Interface')

frame = Frame(window)
options = [' 1)    Create a profile to test its survival'
           ,' 2)    Pick an existing profile to predict survival'
           ,' 3)    View machine learning model\'s rating of features by importance to survival'
           ,' 4)    View Train Data',' 5)   View Test Data'
           ,' 6)    Generate all test predictions', '7)     Quit']
listbox = Listbox(frame, width = 70)
listbox.insert(1, options[0])
listbox.insert(2, options[1])
listbox.insert(3, options[2])
listbox.insert(4, options[3])
listbox.insert(5, options[4])
listbox.insert(6, options[5])
listbox.insert(7, options[6])

selection = ''

def dialog():
    box.askyesno('Selection' , 'Your Choice: ' + \
    listbox.get(listbox.curselection()))
    global selection
    selection = listbox.get(listbox.curselection())
    print(selection)

btn = Button(frame, text = 'Choose', command = dialog)
btn.pack(side = RIGHT, padx = 5)
listbox.pack(side = LEFT)
frame.pack(padx = 30, pady = 30)

window.mainloop()
