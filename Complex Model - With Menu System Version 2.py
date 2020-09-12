#This is where I am importing all of the necessary modules required for this program
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import csv
from time import sleep
import matplotlib.pyplot as plt
from xgboost import plot_importance
import xgboost as xgb

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
    
#This function extracts the data form the csv file and preprocesses it, using one hot encoding
#and imputation, returning it for fitting
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

#This function utilises a temp csv file to import user data to generate a second
#dataframe for panadas to use and make a prediction on
def modelfit_fromsave():
    file_path = 'temp.csv'
    test_data = pd.read_csv(file_path)
    train_data = preprocessor_train()
    train_X = train_data[0]
    valid_X = train_data[1]
    train_y = train_data[2]
    valid_y = train_data[3]
    model = XGBRegressor(max_depth = 5, n_estimators = 1000, learning_rate = 0.19, objective = 'reg:squarederror')
    model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set=[(valid_X, valid_y)], verbose = False)
    predicted_values = model.predict(test_data)
    return predicted_values

#This function generates test predictions
def modelfit_test():
    train_data = preprocessor_train()
    train_X = train_data[0]
    valid_X = train_data[1]
    train_y = train_data[2]
    valid_y = train_data[3]
    test_data = preprocessor_test()
    model = XGBRegressor(max_depth = 5, n_estimators = 1000, learning_rate = 0.19, objective = 'reg:squarederror')
    model.fit(train_X, train_y, early_stopping_rounds = 5, eval_set=[(valid_X, valid_y)], verbose = False)
    predicted_values = model.predict(test_data)
    rounded_values = []
    for item in predicted_values:
        item = round(item,0)
        rounded_values.append(item)
    return rounded_values


#This section is to output the user interface for the program
def menu():
    print('Welcome to the Titanic Survival Menu Interface')
    print('1:Create a profile to test its survival')
    print('2:Pick an existing profile to predict survival')
    print('3:View machine learning model\'s rating of features by importance to survival')
    print('4:View Train Data')
    print('5:View Test Data')
    print('6:Generate all test predictions')
    print('7:Quit')

#This function is essentially an interview, asking for features that it will write to 
#a csv file for the model to fit to later on. This code is longer as I have included validation
def profile_creation_and_test():
   print('''Welcome to Profile Creation. Here you will be able to create your own profile with attributes 
         to find the predicted probability of survival for the profile.''')
   pclass = 'Hello'
   while pclass.isalpha() == True or pclass == '' or pclass not in ['1','2','3']:
       pclass = input('Enter the subject\'s ticket class - (1 = 1st, 2 = 2nd, 3 = 3rd): ') 
   pclass = int(pclass)
   sex = 'Hello'
   while sex.isalpha() == False  or sex == '' or sex not in['m', 'f']:
       sex = input('Enter the subject\'s sex - (m or f): ').lower()
   if sex == 'm':
       sex_male = 1
       sex_female = 0
   elif sex =='f':
       sex_female = 1
       sex_male = 0
   age = 'Hello'
   while age.isalpha() == True  or age == '' or int(age) not in range(1,101):
       age = input('Enter the subject\'s age - (0-100): ')
   age = int(age)
   sibsp = 'Hello'
   while sibsp.isalpha() == True  or sibsp == '' or int(sibsp) not in range(0,16):
       sibsp = input('Enter the subject\'s number of siblings and spouses abord - (0-15):')
   sibsp = int(sibsp)
   parch = 'Hello'
   while parch.isalpha() == True or parch == '' or int(parch) not in range(0,16):
       parch = input('Enter the subject\'s number of parents and children abord - (0-15):')
   parch = int(parch)
   fare = 'Hello'
   while fare.isalpha() == True or fare == '' or int(fare) <= 0:
       fare = input('''Enter the subject\'s passenger fare; in 1912, £100 would be worth over £11000 today
       (typical fares- for 1st class: £50+ , for 2nd class: £20-£50 , for 3rd class: £20 and below): ''')
   fare = int(fare)
   embarked = 'Hello'
   while embarked.isalpha() == False or embarked == '' or embarked not in ['C', 'Q', 'S']:
       embarked = input('Enter the subject\'s port of boarding : C = Cherbourg, Q = Queenstown, S = Southampton: ').upper()
   if embarked == 'S':
      embarked_s = 1
      embarked_c = 0
      embarked_q = 0
   if embarked == 'C':
      embarked_c = 1
      embarked_s = 0
      embarked_q = 0
   if embarked == 'Q':
      embarked_q = 1
      embarked_c = 0
      embarked_s = 0
   profile = (pclass, age, sibsp, parch, fare, sex_female, sex_male, embarked_c, embarked_q, embarked_s)
   #This section writes to the temp file; it also creates a temp file if it doesn't exist
   file = open('temp.csv', 'w+')
   with file:
      write = csv.writer(file)
      headers = ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S')
      write.writerow(headers)
      write.writerow(profile)
   sleep(1)
   print('Predicted probability of survival =', modelfit_fromsave())
   sleep(5)

#This variable allows option 7 to quit the program
quitter = True   

#This is the loop that runs that main section of the code. It calls the functions defined earlier
while quitter == True:
    menu()
    response = input('Press the designated number to select what you would like to do: ')
    if response == '1':
        profile_creation_and_test()
        continue
    elif response == '2':
        answer = 'Hello'
        while answer.isalpha() == True or answer == '' or int(answer) not in range(1,419):
            answer = input('There are 418 profiles to select from. Please enter a number between 1 and 418:')
        answer = int(answer)
        index = answer-1
        answer = answer+1
        file = open('test.csv', 'r')
        with file:
            read = csv.reader(file)
            for i in range(1, answer):
                next(read)
            row = next(read)
        print('You have chosen: ', row[2])
        sleep(2)
        print('Class: ', row[1], '     Sex: ',row[3], '     Age: ', row[4], '     Siblings/Spouses: ',row[5])
        sleep(2)
        print('Parents/Children: ', row[6], '     Fare: ', row[8], '     Embarked', row[10])
        predictions = modelfit_test()
        sleep(3)
        print('Predicted probability of survival: ', predictions[index])
        sleep(5)
        continue
    elif response == '3':
        train_data = preprocessor_train()
        train_X = train_data[0]
        train_y = train_data[2]
        model = xgb.XGBClassifier()
        model.fit(train_X,train_y)
        #This section plots feature importance
        sleep(2)
        print('Plotting...')
        sleep(2)
        plot_importance(model)
        plt.show()
        sleep(5)
        continue
    elif response == '4':
        #Outputs train data
        file = open('train.csv', 'r')
        for line in file:
            print(file.read())
        sleep(5)
        continue
    elif response == '5':
        #Outputs test data
        file = open('test.csv', 'r')
        for line in file:
            print(file.read())
        sleep(5)
        continue
    elif response == '6':
        #Outputs test predictions
        print(modelfit_test())
        sleep(5)
        continue
    elif response == '7':
        quitter = False
    else:
        print('Invalid input')
