#Note this time I import the xgboost module for the complex model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

file_path = 'train.csv'
data = pd.read_csv(file_path)
data.Fare = data.Fare.astype(int)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[features]
y = data.Survived

#Split data into train and test segments
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state = 0)

#Here is the data preprocessing code from my cleaner with some further automation modifications
num_X_train = train_X.drop(['Sex', 'Embarked'], axis=1)
num_X_valid = valid_X.drop(['Sex', 'Embarked'], axis=1)

imputer = SimpleImputer(strategy = 'most_frequent')
imputed_X_train = pd.DataFrame(imputer.fit_transform(num_X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(num_X_valid))

imputed_X_train.columns = num_X_train.columns
imputed_X_valid.columns = num_X_valid.columns

s = (train_X.dtypes == 'object')
object_columns = list(s[s].index)

train_X_plus = train_X[object_columns].copy()
valid_X_plus = valid_X[object_columns].copy()

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
    
    
train_X_plus['Embarked'] = train_X_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
valid_X_plus['Embarked'] = valid_X_plus['Embarked'].fillna(value = impute_embarked, axis = 0)
train_X_plus['Sex'] = train_X_plus['Sex'].fillna(value = impute_sex, axis = 0)
valid_X_plus['Sex'] = valid_X_plus['Sex'].fillna(value = impute_sex, axis = 0)

OH_cols_train = pd.get_dummies(train_X_plus)
OH_cols_valid = pd.get_dummies(valid_X_plus)


processed_X_train = pd.concat([imputed_X_train.reset_index(), OH_cols_train.reset_index()], axis=1)
processed_X_valid = pd.concat([imputed_X_valid.reset_index(), OH_cols_valid.reset_index()], axis=1)

processed_X_train = processed_X_train.drop(['index'], axis =1)
processed_X_valid = processed_X_valid.drop(['index'], axis =1)

processed_X_train = processed_X_train.apply(pd.to_numeric)
processed_X_valid = processed_X_valid.apply(pd.to_numeric)

#The complex model
model = XGBRegressor(max_depth = 5, n_estimators = n_estimators, learning_rate = learning_rate,
                         objective = 'reg:squarederror')
model.fit(processed_X_train, train_y, early_stopping_rounds = early_stopping_rounds,
                          eval_set=[(processed_X_valid, valid_y)], verbose = False)
predicted_values = model.predict(processed_X_valid)
rounded_values = []
for item in predicted_values:
    item = round(item, 0)
    rounded_values.append(item)
accuracy = (accuracy_score(rounded_values,valid_y)*100)