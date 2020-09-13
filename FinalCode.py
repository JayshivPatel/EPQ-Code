import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import csv
from time import sleep
import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb
from tkinter import *
import tkinter.messagebox as box


#Fonts:
blarge = 'Helvetica 13 bold'
large = 'Helvetica 13'
bold = 'Helvetic 11 bold'



#The first set of major subroutines are all related to the GUI
#These screens are the main menu and selection respectively (Screen 1 and 2)
def Menu():
	window = Tk()
	window.geometry('640x360')
	window.title('Titanic Survival Menu Interface')
	instruction = Label(window, text = "Choose an option from the list below, then press 'Select'", font = blarge)
	frame = Frame(window)
	options = [' 1) Create a profile to test its survival'
	,' 2) Pick an existing profile to predict survival'
	,' 3) View machine learning model\'s rating of features by importance to survival'
	,' 4) View Train Data',' 5) View Test Data'
	,' 6) Generate all test predictions', ' 7) View Key for terms referenced', ' 8) Quit']


	#Putting the entires into a listbox for the user to select from
	listbox = Listbox(frame, width = 70)
	listbox.insert(1, options[0])
	listbox.insert(2, options[1])
	listbox.insert(3, options[2])
	listbox.insert(4, options[3])
	listbox.insert(5, options[4])
	listbox.insert(6, options[5])
	listbox.insert(7, options[6])
	listbox.insert(8, options[7])
	selection = ''


	#This subroutine creates a selection confirmation screen
	def dialog():
		var = box.askyesno('Selection' , 'Your Choice: ' + listbox.get(listbox.curselection()).strip('1234567)'))
		global selection
		if var == 1:
			selection = listbox.get(listbox.curselection())
				
			window.destroy()


	#Packing features
	btn = Button(frame, text = 'SELECT', command = dialog, font = blarge)
	instruction.pack(side = TOP)
	btn.pack(side = RIGHT, padx = 5)
	listbox.pack(side = LEFT)
	frame.pack(padx = 30, pady = 30)

	window.mainloop()


#Screen 4
def ProfileCreationandTest():
	#This allows the program to detect when all inputs are valid
	validate = [False,False,False,False,False,False,False]


	#These are all validation subroutines that check the validity of entries
	#They disable entires once they are acceptable
	def check_pclass():
	    
	    pclass = entry1.get()
	    
	    if pclass.isalpha() == True or pclass == '' or pclass not in ['1','2','3']:
	        entry1.configure(fg = 'red')        
	    
	    else:
	        btn1.configure(fg = 'green')
	        entry1.config(state = DISABLED)
	        validate[0] = True


	def check_sex():
	    
	    sex = entry2.get()
	    
	    if sex.isalpha() == False  or sex == '' or sex not in['m', 'f']:
	        entry2.configure(fg = 'red')
	    
	    else:
	        btn2.configure(fg = 'green')
	        entry2.config(state = DISABLED)
	        validate[1] = True


	def check_age():
	    
	    age = entry3.get()
	    
	    if age.isalpha() == True  or age == '' or int(age) not in range(1,101):
	        entry3.configure(fg = 'red')
	    
	    else:
	        btn3.configure(fg = 'green')
	        entry3.config(state = DISABLED)
	        validate[2] = True


	def check_sibsp():
	    
	    sibsp = entry4.get()
	   
	    if sibsp.isalpha() == True  or sibsp == '' or int(sibsp) not in range(0,16):
	        entry4.configure(fg = 'red')
	    
	    else:
	        btn4.configure(fg = 'green')
	        entry4.config(state = DISABLED)
	        validate[3] = True


	def check_parch():
	    
	    parch = entry5.get()
	    
	    if parch.isalpha() == True or parch == '' or int(parch) not in range(0,16):
	        entry5.configure(fg = 'red')
	    
	    else:
	        btn5.configure(fg = 'green')
	        entry5.config(state = DISABLED)
	        validate[4] = True


	def check_fare():
	    
	    fare = entry6.get()
	    
	    if fare.isalpha() == True or fare == '' or int(fare) <= 0:
	        entry6.configure(fg = 'red')
	    
	    else:
	        btn6.configure(fg = 'green')
	        entry6.config(state = DISABLED)
	        validate[5] = True


	def check_embarked():
	    
	    embarked = entry7.get().lower()
	    
	    if embarked.isalpha() == False or embarked == '' or embarked not in ['c', 'q', 's']:
	        entry7.configure(fg = 'red')
	    
	    else:
	        btn7.configure(fg = 'green')
	        entry7.config(state = DISABLED)
	        validate[6] = True


	#This subroutine checks that all inputs have been previously checked.
	def check_all():
	    
	    check = True
	    
	    for item in validate:
	        if item == False:
	            check = False
	    
	    if check == True:
	        profile = (entry1.get(),entry2.get(),entry3.get(),entry4.get(),
	                   entry5.get(),entry6.get(),entry7.get())
	        predictor(profile)
	    
	    else:
	        btn8.configure(fg = 'red')


	#This checks probability of survival
	def predictor(profile):
	    
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


	#Writing to the temporary file to predict from it later
	    file = open('temp.csv', 'w+')
	    
	    with file:
	        write = csv.writer(file)
	        headers = ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S')
	        write.writerow(headers)
	        write.writerow(updatedprofile)

	    prediction = modelfit_fromsave()
	    prediction = float(prediction)
	    prediction = round(prediction, 4)
	    concatanate = ('Probability of Survival:'+ str(prediction))
	#Screen 3
	    box.showinfo('Prediction', concatanate)
	    
	    window.destroy()


	#This contains the window features, each variable is a different feature
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

	mainlabel = Label(window, text = '''Welcome to the Profile Check.
	    Here you will be able to create your own profile with attributes 
	    to find the predicted probability of survival for the profile.
	    Type in attributes, and check each feature before predicting
	    using the buttons on the side and below:''', font = large)
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
	#These gaps have been added to help organise the other features    
	gap1 = Label(window, text = '')
	gap2 = Label(window, text = '')
	gap3= Label(window, text = '')

	btn1 = Button(window,text = 'CHECK', command = check_pclass, font = bold)
	btn2 = Button(window,text = 'CHECK', command = check_sex, font = bold)
	btn3 = Button(window,text = 'CHECK', command = check_age, font = bold)
	btn4 = Button(window,text = 'CHECK', command = check_sibsp, font = bold)
	btn5 = Button(window,text = 'CHECK', command = check_parch, font = bold)
	btn6 = Button(window,text = 'CHECK', command = check_fare, font = bold)
	btn7 = Button(window,text = 'CHECK', command = check_embarked, font = bold)
	btn8 = Button(window,text = 'PREDICT...', command = check_all, font = bold)

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


#Screen5
def ProfileSelectionandTest():
    
	def check_answer():

		global answer 
		answer = entry1.get()

		if answer.isalpha() == True or answer == '' or int(answer) not in range(1,419):
		    entry1.configure(fg = 'red')        

		else:
			entry1.configure(fg = 'green')
			global index
			answer = int(answer)
			index = answer-1
			answer = answer+1

		fromcsv()
	#This outputs the profile and predicted survival
	def fromcsv():

		file = open('test.csv', 'r')

		with file:
		    read = csv.reader(file)
			
		    for i in range(1, answer):
			    next(read)
			    row = next(read)

		var1 = 'You have chosen: ' + str(row[2])
		var2 = 'Class: ' + str(row[1])
		var3 = 'Sex: ' + str(row[3])
		var4 = 'Age: ' + str(row[4])
		var5 = 'Siblings/Spouses: ' + str(row[5])
		var6 = 'Parents/Children: ' + str(row[6])
		var7 = 'Fare: ' + str(row[8])
		var8 = 'Embarked: ' + str(row[10])

		label3 = Label(window, text = var1.strip("''"))
		label4 = Label(window, text = var2.strip("''"))
		label5 = Label(window, text = var3.strip("''"))
		label6 = Label(window, text = var4.strip("''"))
		label7 = Label(window, text = var5.strip("''"))
		label8 = Label(window, text = var6.strip("''"))
		label9 = Label(window, text = var7.strip("''"))
		label10 = Label(window, text = var8.strip("''"))

		label2.grid(row = 5, column = 1)
		window.update()
		sleep(2)
		label3.grid(row = 6, column = 1, pady = 10)
		label4.grid(row = 7, column = 1, pady = 10)
		label5.grid(row = 8, column = 1, pady = 10)
		label6.grid(row = 9, column = 1, pady = 10)
		label7.grid(row = 10, column = 1, pady = 10)
		label8.grid(row = 6, column = 2, pady = 10)
		label9.grid(row = 7, column = 2, pady = 10)
		label10.grid(row = 8, column = 2, pady = 10)

		window.update()

		sleep(3)

		predict()

	def predict():

		predictions = modelfit_test()
		prediction = predictions[index]
		prediction = round(prediction, 4)
		concatanate = 'Probability of Survival:'+str(prediction)
		#Screen 3
		box.showinfo('Prediction', concatanate)

		window.destroy()

	window = Tk()
	window.geometry('720x405')
	window.title('Predicting survival of internal profile')

	label1 = Label(window, text = "There are 418 profiles to select from. \n Please enter a number between 1 and 418, \n then press 'Predict'")
	entry1 = Entry(window)
	btn1 = Button(window, text = 'PREDICT', command = check_answer, font = blarge)
	label2 = Label(window, text = 'You have chosen:', font = blarge)
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


#Screen 6
def FeatureImportance():
#Because each window is separate it needs its own back function
	def goback():
		window.destroy()

	window = Tk()    
	window.geometry('960x540')
	window.title('Feature Importance')

	label1 = Label(window, text = 'Here you can view the  machine learning model\'s rating of features by importance to survival \n x-axis = importance(F-score), y-axis = features')
	graph = PhotoImage(file = 'graph.gif')
	label2 = Label(window, image = graph)    
	btn1 = Button(window, text = 'BACK', command = goback, font = blarge)

	label1.pack()
	label2.pack()
	btn1.pack()

	window.mainloop()
    

#Screen 7
def TrainData():
    
	def goback():
		window.destroy()

	window = Tk()
	window.geometry('640x360')
	window.title('Train Data')
	frame = Frame(window)

	#There is a large amount of data and so a scrollbar is needed here
	scrollbar = Scrollbar(frame, orient = VERTICAL)
	listbox = Listbox(frame, width = 90, yscrollcommand = scrollbar.set)
	count = 1
	file = open('train.csv', 'r')

	for line in file:
		listbox.insert(count, line)
		count+=1

	scrollbar.pack(side=RIGHT, fill=Y)
	btn1 = Button(window, text = 'BACK', command = goback, font = blarge)

	scrollbar.config(command = listbox.yview)
	scrollbar.pack(side=RIGHT, fill = Y)
	listbox.pack()
	frame.pack(padx = 30, pady = 30)
	btn1.pack(side= BOTTOM)

	window.mainloop() 

#Screen 8
def TestData():
    
	def goback():
		window.destroy()

	window = Tk()
	window.geometry('640x360')
	window.title('Test Data')
	frame = Frame(window)  

	scrollbar = Scrollbar(frame, orient = VERTICAL)
	listbox = Listbox(frame, width = 90, yscrollcommand = scrollbar.set)
	count = 1

	file = open('test.csv', 'r')
	for line in file:
                listbox.insert(count, line)
                count+=1

	scrollbar.pack(side=RIGHT, fill=Y)
	btn1 = Button(window, text = 'BACK', command = goback, font = blarge)

	scrollbar.config(command = listbox.yview)
	scrollbar.pack(side=RIGHT, fill = Y)
	listbox.pack()
	frame.pack(padx = 30, pady = 30)
	btn1.pack(side= BOTTOM)

	window.mainloop()

#Screen 9 
def TestPredictions():
	def goback():
		window.destroy()
	window = Tk()
	window.geometry('480x270')
	window.title('Test Predictions')
	frame = Frame(window)

	scrollbar = Scrollbar(frame, orient = VERTICAL)
	listbox = Listbox(frame, width = 70, yscrollcommand = scrollbar.set)

	predictions = modelfit_test()
	rounded_values = []

	for item in predictions:
		item = round(item,4)
		rounded_values.append(item)

	count = 1

	for item in rounded_values:
		label = str(count)+ ': '+str(item)
		listbox.insert(count, label)
		count+=1

	scrollbar.pack(side=RIGHT, fill=Y)
	btn1 = Button(window, text = 'BACK', command = goback, font = blarge)

	scrollbar.config(command = listbox.yview)
	scrollbar.pack(side=RIGHT, fill = Y)
	listbox.pack()
	frame.pack(padx = 30, pady = 30)
	btn1.pack(side= BOTTOM)

	window.mainloop()

#Screen 10
def Key():
	def goback():
		window.destroy()

	window = Tk()
	window.geometry('1200x675')
	window.title('Key')

	key = PhotoImage(file = 'key.gif')
	label2 = Label(window, image = key)
	btn1 = Button(window, text = 'BACK', command = goback, font = blarge)

	label2.pack()
	btn1.pack()

	window.mainloop()

#These two functions are required to imitate the most frequent imputation method with categorical data,
#port and sex
def imputation_embarked(column):
    
	#They operate similarly, with one counting and returning the most frequent sex,and the other the port.
	S = 0
	C = 0
	Q = 0

	for item in column:

		if item == 'S':
		    S+=1

		if item == 'C':
		    C+=1

		if item == 'Q':
		    Q+=1

	embarked = [S,C,Q]
	max_embarked = max(embarked)

	if max_embarked == S:    
	    impute_embarked = 'S'

	if max_embarked == C:
	    impute_embarked = 'C'

	else:
	    impute_embarked = 'Q'

	return impute_embarked

def imputation_sex(column):    
    
	male = 0
	female = 0

	for item in column:

		if item == 'male':
		    male+=1

		if item == 'female':
		    female+=1

	sex = [male, female]
	max_sex = max(sex)

	if max_sex == male:
	    impute_sex = 'male'

	else:
	    impute_sex = 'female'

	return impute_sex

#This function is the 'datacleaner'. It performs the rest of the data manipulation
def impute_and_ohencoder(data):

	#First we remove any categorical data - this has already been cleaned
	num_data = data.drop(['Sex', 'Embarked'], axis=1)
	imputer = SimpleImputer(strategy = 'most_frequent')
	imputed_data = pd.DataFrame(imputer.fit_transform(num_data))

	# Imputation removed column names; put them back
	imputed_data.columns = num_data.columns
	imputed_data.columns = num_data.columns

	# Getting list of categorical variables
	s = (data.dtypes == 'object')

	# Making copy to avoid changing original data (when imputing)
	object_columns = list(s[s].index)
	data_plus = data[object_columns].copy()

	#Fill in any gaps because we used the most frequent imputer earlier I am copying the behaviour here   
	impute_embarked = imputation_embarked(data_plus['Embarked'])
	impute_sex = imputation_sex(data_plus['Sex'])

	# Apply one-hot encoder to each column with categorical data
	data_plus['Embarked'] = data_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
	data_plus['Sex'] = data_plus['Sex'].fillna(value = impute_sex, axis = 0)
	OH_cols = pd.get_dummies(data_plus)

	# Add one-hot encoded columns to numerical features
	processed_data= pd.concat([imputed_data.reset_index(), OH_cols.reset_index()], axis=1)
	processed_data = processed_data.drop(['index'], axis =1)
	processed_data = processed_data.apply(pd.to_numeric)   

	return(processed_data)
    
#This function extracts the data form the csv file
def preprocessor_train():
    
	file_path = 'train.csv'
	data = pd.read_csv(file_path)

	#I will have to remove the name field as this is irrelevant, the ticket field as this contains both letters and 
	#numbers and the cabin field as the number of missing entries is high. 
	features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	X = data[features]
	y = data.Survived

	#Split data into train and test segments
	train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 0)
	processed_X_train = impute_and_ohencoder(train_X)
	processed_X_valid = impute_and_ohencoder(valid_X)

	return [processed_X_train,processed_X_valid, train_y, valid_y]

#This function is similar to the previous one, except extracts data from the test file.
def preprocessor_test():
    
	file_path = 'test.csv'
	data = pd.read_csv(file_path)
	features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
	X = data[features]

	processed_X = impute_and_ohencoder(X)

	return(processed_X)

def modelfit(test_data):
    
	train_data = preprocessor_train()
	train_X = train_data[0]
	valid_X = train_data[1]
	train_y = train_data[2]
	valid_y = train_data[3]

	model = XGBRegressor(max_depth = 5, n_estimators = 1000, learning_rate = 0.19, objective = 'reg:squarederror')
	model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set=[(valid_X, valid_y)], verbose = False)
	predicted_values = model.predict(test_data)
    
	return predicted_values
#This function utilises a temp csv file to import user data to generate a second
#dataframe for panadas to use and make a prediction on
def modelfit_fromsave():
    
	file_path = 'temp.csv'
	test_data = pd.read_csv(file_path)

	return (modelfit(test_data))

#This function generates test predictions
def modelfit_test():
    
	train_data = preprocessor_train()
	test_data = preprocessor_test()

	return (modelfit(test_data))


#Allows option 8 to quit
quitter = False


#Main program
while quitter == False:

	Menu()

	decider = ''

	for character in selection:
	    if character.isdigit() == True:
			decider = character

	if decider == '1':
		ProfileCreationandTest()
	    
		continue

	elif decider == '2':
		ProfileSelectionandTest()
	    
		continue

	elif decider == '3': 
		FeatureImportance()
	    
		continue

	elif decider == '4':
		TrainData()
	    
		continue

	elif decider == '5':
		TestData()
	    
		continue

	elif decider == '6':
		TestPredictions()
	    
		continue

	elif decider == '7':
		Key()
	    
		continue

	else:
		quitter = True
	    
