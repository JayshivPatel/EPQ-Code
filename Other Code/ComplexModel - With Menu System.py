import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import csv
from time import sleep
import matplotlib.pyplot as plt

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
    
    #Here is the data preprocessing code from my cleaner with some automation modifications
    #First we remove any categorical data - this will be sorted later and isn't missing many entries
    num_X_train = train_X.drop(['Sex', 'Embarked'], axis=1)
    num_X_valid = valid_X.drop(['Sex', 'Embarked'], axis=1)

    imputer = SimpleImputer(strategy = 'most_frequent')
    imputed_X_train = pd.DataFrame(imputer.fit_transform(num_X_train))
    imputed_X_valid = pd.DataFrame(imputer.transform(num_X_valid))

    # Imputation removed column names; put them back
    imputed_X_train.columns = num_X_train.columns
    imputed_X_valid.columns = num_X_valid.columns
    # Get list of categorical variables
    s = (train_X.dtypes == 'object')
    object_columns = list(s[s].index)
    # Make copy to avoid changing original data (when imputing)
    train_X_plus = train_X[object_columns].copy()
    valid_X_plus = valid_X[object_columns].copy()
    #Here I am using for loops to imitiate the imputation for categorical data. This cannot be done using the imputer
    #itself as this only works for numerical data.
    S = 0
    C = 0
    Q = 0
    for item in train_X_plus['Embarked']:
        if item == 'S':
            S+=1
        if item == 'C':
            C+=1
        if item == 'Q':
            Q+=1
    for item in valid_X_plus['Embarked']:
        if item == 'S':
            S+=1
        if item == 'C':
            C+=1
        if item == 'Q':
            Q+=1
    male = 0
    female = 0

    for item in train_X_plus['Sex']:
        if item == 'male':
            male+=1
        if item == 'female':
            female+=1
    for item in valid_X_plus['Sex']:
        if item == 'male':
            male+=1
        if item == 'female':
            female+=1
    
    sex = [male, female]
    embarked = [S,C,Q]
    max_sex = max(sex)
    max_embarked = max(embarked)

    if max_sex == male:
        impute_sex = 'male'
    else:
        impute_sex = 'female'
    
    if max_embarked == S:
        impute_embarked = 'S'
    if max_embarked == C:
        impute_embarked = 'C'
    else:
        impute_embarked = 'Q'
    
    #Fill in any gaps because we used the most frequent imputer earlier I will imitate the behaviour here    
    train_X_plus['Embarked'] = train_X_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
    valid_X_plus['Embarked'] = valid_X_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
    train_X_plus['Sex'] = train_X_plus['Sex'].fillna(value = impute_sex, axis = 0)
    valid_X_plus['Sex'] = valid_X_plus['Sex'].fillna(value = impute_sex, axis = 0)
    # Apply one-hot encoder to each column with categorical data
    OH_cols_train = pd.get_dummies(train_X_plus)
    OH_cols_valid = pd.get_dummies(valid_X_plus)
    # Add one-hot encoded columns to numerical features
    processed_X_train = pd.concat([imputed_X_train.reset_index(), OH_cols_train.reset_index()], axis=1)
    processed_X_valid = pd.concat([imputed_X_valid.reset_index(), OH_cols_valid.reset_index()], axis=1)

    processed_X_train = processed_X_train.drop(['index'], axis =1)
    processed_X_valid = processed_X_valid.drop(['index'], axis =1)

    processed_X_train = processed_X_train.apply(pd.to_numeric)
    processed_X_valid = processed_X_valid.apply(pd.to_numeric)
    
    return [processed_X_train,processed_X_valid, train_y, valid_y]
    
def preprocessor_test():
    file_path = 'test.csv'
    data = pd.read_csv(file_path)
    #I will have to remove the name field as this is irrelevant, the ticket field as this contains both letters and 
    #numbers and the cabin field as the number of missing entries is high. 
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = data[features]
    #Here is the data preprocessing code from my cleaner with some automation modifications
    #First we remove any categorical data - this will be sorted later and isn't missing many entries
    num_X = X.drop(['Sex', 'Embarked'], axis=1)

    imputer = SimpleImputer(strategy = 'most_frequent')
    imputed_X = pd.DataFrame(imputer.fit_transform(num_X))

    # Imputation removed column names; put them back
    imputed_X.columns = num_X.columns
    imputed_X.columns = num_X.columns
    # Get list of categorical variables
    s = (X.dtypes == 'object')
    object_columns = list(s[s].index)
    # Make copy to avoid changing original data (when imputing)
    X_plus = X[object_columns].copy()
    #Here I am using for loops to imitiate the imputation for categorical data. This cannot be done using the imputer
    #itself as this only works for numerical data.
    S = 0
    C = 0
    Q = 0
    for item in X_plus['Embarked']:
        if item == 'S':
            S+=1
        if item == 'C':
            C+=1
        if item == 'Q':
            Q+=1

    male = 0
    female = 0

    for item in X_plus['Sex']:
        if item == 'male':
            male+=1
        if item == 'female':
            female+=1
    sex = [male, female]
    embarked = [S,C,Q]

    max_sex = max(sex)
    max_embarked = max(embarked)

    if max_sex == male:
        impute_sex = 'male'
    else:
        impute_sex = 'female'
    
    if max_embarked == S:
        impute_embarked = 'S'
    if max_embarked == C:
        impute_embarked = 'C'
    else:
        impute_embarked = 'Q'
    
    #Fill in any gaps because we used the most frequent imputer earlier I will imitate the behaviour here    
    X_plus['Embarked'] = X_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
    X_plus['Sex'] = X_plus['Sex'].fillna(value = impute_sex, axis = 0)
    # Apply one-hot encoder to each column with categorical data
    OH_cols = pd.get_dummies(X_plus)
    # Add one-hot encoded columns to numerical features
    processed_X= pd.concat([imputed_X.reset_index(), OH_cols.reset_index()], axis=1)

    processed_X = processed_X.drop(['index'], axis =1)

    processed_X = processed_X.apply(pd.to_numeric)
    
    return processed_X

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


#This section is to create the user interface for the system
def menu():
    print('Welcome to the Titanic Survival Menu Interface')
    print('1:Create a profile to test its survival')
    print('2:Pick an existing profile to predict survival')
    print('3:View machine learning model\'s rating of features by importance to survival')
    print('4:View Train Data')
    print('5:View Test Data')
    print('6:Generate all test predictions')
    print('7:Quit')

def profile_creation_and_test():
   print('''Welcome to Profile Creation. Here you will be able to create your own profile with attributes 
         to find the predicted probability of survival for the profile.''')
   pclass = 'Hello'
   while pclass.isalpha() == True or pclass == '' or pclass not in ['1','2','3']:
       pclass = input('Enter the subject\'s ticket class - (1 = 1st, 2 = 2nd, 3 = 3rd): ') 
   pclass = int(pclass)
   sex = 'Hello'
   while sex.isalpha() == False  or sex == '' or sex not in['m', 'f']:
       sex = input('Enter the subject\'s sex - (m or f): ')
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
       sibsp = input('Enter the subject\'s number of siblings and spouses abord - (0-15) ')
   sibsp = int(sibsp)
   parch = 'Hello'
   while parch.isalpha() == True or parch == '' or int(parch) not in range(0,16):
       parch = input('Enter the subject\'s number of parents and children abord - (0-15)')
   parch = int(parch)
   fare = 'Hello'
   while fare.isalpha() == True or fare == '' or int(fare) <= 0:
       fare = input('''Enter the subject\'s passenger fare; in 1912, £100 would be worth over £11000 today 
                    (typical fares- for 1st class: £50+ , for 2nd class: £20-£50 , for 3rd class: £20 and below): ''')
   fare = int(fare)
   embarked = 'Hello'
   while embarked.isalpha() == False or embarked == '' or embarked not in ['C', 'Q', 'S']:
       embarked = input('Enter the subject\'s port of boarding : C = Cherbourg, Q = Queenstown, S = Southampton: ')
   embarked = embarked.upper()
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
   file = open('temp.csv', 'w+')
   with file:
      write = csv.writer(file)
      headers = ('Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S')
      write.writerow(headers)
      write.writerow(profile)
   sleep(1)
   print('Predicted probability of survival =', modelfit_fromsave())
   sleep(5)


quitter = True   

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
        file = open('../input/test.csv', 'r')
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
        from xgboost import plot_importance
        import xgboost as xgb
        train_data = preprocessor_train()
        train_X = train_data[0]
        train_y = train_data[2]
        model = xgb.XGBClassifier()
        model.fit(train_X,train_y)
        #plot feature importance
        sleep(2)
        print('Plotting...')
        sleep(2)
        plot_importance(model)
        plt.show()
        sleep(5)
        continue
    elif response == '4':
        file = open('train.csv', 'r')
        for line in file:
            print(file.read())
        sleep(5)
        continue
    elif response == '5':
        file = open('test.csv', 'r')
        for line in file:
            print(file.read())
        sleep(5)
        continue
    elif response == '6':
        print(modelfit_test())
        sleep(5)
        continue
    elif response == '7':
        quitter = False
    else:
        print('Invalid input')
