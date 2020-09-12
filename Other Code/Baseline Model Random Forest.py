#This section allows me to import the modules needed for this baseline model, most 
#importantly the Random Forest Module
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# This section allows me to import the modified csv file with the data. I manually inputted 
#1/0 for male/female and followed a similar procedure for the port of embarkation. For 
#simplicity on the first model, I have used the function dropna in order to drop entries 
#with missing values.

file_path = 'train_modified.csv'
data = pd.read_csv(file_path)
data = data.dropna(axis=0)

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

#This allows me to define my model. The random state is good practice 
#to ensure the same result each run.
baseline_model = RandomForestRegressor(random_state = 1)
#Fiting model; the output that follows also includes the default settings.
baseline_model.fit(X,y)
#Predicting and outputting values
predicted_values = baseline_model.predict(X)
print(predicted_values)


