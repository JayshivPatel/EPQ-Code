#This section allows me to import the modules needed for this baseline model, most 
#importantly the Random Forest Module
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

file_path = 'train.csv'
data = pd.read_csv(file_path)
#Ths rounds the floats to integers
data.Fare = data.Fare.astype(int)

#This selects my prediction target, y
y = data.Survived
#These are the feautures which will be used by the model in order 
#to determine survival. Name has been excluded.
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
#Data is called X - by convention
X = data[features]
#This outputs a quick statistical review of the data. (Unecessary in raw code - useful for
#display in the notebook)
#X.describe()

#Here I am going to split the data into test/train sections to give an 
#idea of accuracy. The random state is good practice 
#to ensure the same result each run.
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 1)

#First we remove any categorical data - this will be sorted later and isn't missing many entries
num_X_train = train_X.drop(['Sex', 'Embarked'], axis=1)
num_X_valid = valid_X.drop(['Sex', 'Embarked'], axis=1)

#Here I am going to impute (fill in missing gaps) using the most frequent occurances in the dataset.
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

#Here I am using for loops to imitiate the imputation for categorical data. 
#This cannot be done using the imputer itself as this only works for numerical data.
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
#print(S,C,Q)

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
#print(male,female)

#Fill in any gaps using the mode because we used the most frequent imputer earlier 
#I will imitate the behaviour here

train_X_plus['Embarked'] = train_X_plus['Embarked'].fillna(value = 'S', axis = 0)
valid_X_plus['Embarked'] = valid_X_plus['Embarked'].fillna(value = 'S', axis = 0)
train_X_plus['Sex'] = train_X_plus['Sex'].fillna(value = 'male', axis = 0)
valid_X_plus['Sex'] = valid_X_plus['Sex'].fillna(value = 'male', axis = 0)


# Apply one-hot encoder to each column with categorical data (Sex and Embarked)
OH_cols_train = pd.get_dummies(train_X_plus)
OH_cols_valid = pd.get_dummies(valid_X_plus)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([imputed_X_train.reset_index(), OH_cols_train.reset_index()], axis=1)
OH_X_valid = pd.concat([imputed_X_valid.reset_index(), OH_cols_valid.reset_index()], axis=1)

#This allows me to define my model. 
baseline_model = RandomForestRegressor(random_state = 1, max_leaf_nodes = 8)
#Fiting model
baseline_model.fit(OH_X_train,train_y)
#Predicting and outputting values
predicted_values = baseline_model.predict(OH_X_valid)
print(predicted_values)

#Here to overcome the issue of decimals, I used a rounding function
rounded_values =[]
for item in predicted_values:
    item = round(item, 0)
    rounded_values.append(item)
#This function calculates the accuracy of the model's predictions
print(accuracy_score(rounded_values,valid_y)*100)