from tkinter import *

window = Tk()
window.geometry('960x540')
window.title('Feature Importance')
label1 = Label(window, text = 'Here you can view the  machine learning model\'s rating of features by importance to survival \n x-axis = importance(F-score), y-axis = features')
graph = PhotoImage(file = 'graph.gif')
label2 = Label(window, image = graph)
btn1 = Button(window, text = 'Ok', command = None)
label1.pack()
label2.pack()
btn1.pack()
window.mainloop()
