import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Quikr_car.csv')
#print(df.info())

df = df.drop(['Unnamed: 0'], axis = 'columns')
len(df['Name'].unique())

df['Name'] = df['Name'].str.split(" ")

df['Name'] = df['Name'].str.slice(0, 3).str.join(" ")

#use lambda function

hyphenReplace = lambda x: x.replace('-', '') 
df['Name'] = df['Name'].apply(hyphenReplace)

#NULL CHECK
# for i in range(9):
#     print("The number of null values in column", i, "are: ")
#     nullCheck = df.iloc[: , i].isnull().sum()
#     print(nullCheck)

nullValues_df = df[df['Location'].isnull()]
nullLocation = df[df['Location'].isnull()].index
#print(nullLocation)

df = df.drop(nullLocation)

# NULL CHECK
# for i in range(9):
#     print("The number of null values in column", i, "are: ")
#     nullCheck = df.iloc[: , i].isnull().sum()
#     print(nullCheck)

#remove the rupees symbol and commas in the price column
df['Price'] = df['Price'].str.replace(',' , '')
df['Price'] = df['Price'].str.replace('₹','')

askForPrice = df[df['Price'] == 'Ask For Price'].index
df = df.drop(askForPrice)

#type casting the string type to integer type in the price column
df['Price'] = pd.to_numeric(df['Price'])

#remove the kms symbol and commas in the Kms_driven column
df['Kms_driven'] = df['Kms_driven'].str.replace(',' , '')
df['Kms_driven'] = df['Kms_driven'].str.replace('kms','')

#type casting the string type to integer type in the Kms_driven column
df['Kms_driven'] = pd.to_numeric(df['Kms_driven'])

#Owner column isn't really required and it has many null values
df = df.drop(['Owner'], axis = 'columns')

#changing strings(PLATINUM and GOLD) into specified integer values in the Label column
labels = {'PLATINUM' : 2 , 'GOLD' : 1}
df = df.replace(labels)

#using dummies concept we assign certain values to each location in the Location column

dummies_1 = pd.get_dummies(df['Location'])

df = df.drop(['Location'], axis = 'columns')
df = pd.concat([df , dummies_1], axis = 'columns')

df = df.drop(df.columns[45], axis = 'columns') #standard technique to remove dependency amongst the data as we used dummies concept

dummies_2 = pd.get_dummies(df['Fuel_type'])

#using dummies concept we assign certain values to each location in the Fuel_type column
df = df.drop(['Fuel_type'], axis = 'columns')
df = pd.concat([df , dummies_2], axis = 'columns')
df = df.drop(df.columns[53], axis = 'columns')

#resetting the index
df = df.reset_index(drop = True)

dummies_3 = pd.get_dummies(df['Name'])

dummies_4 = pd.get_dummies(df['Company'])

df = df.drop(['Name' , 'Company'], axis = 'columns')

df = pd.concat([df, dummies_3, dummies_4], axis = 'columns')

df = df.drop(df.columns[55], axis = 'columns')
df = df.drop(df.columns[303], axis = 'columns')

##### DATA PREPROCESSING
# def normalize(feature):
#     feature = feature  - (np.mean(feature) / np.std(feature))
#     return feature

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Kms_driven = np.array(df['Kms_driven'])
Kms_driven = Kms_driven.reshape(-1, 1) #converting 1-D array to 2-D array
df['Kms_driven'] = scaler.fit_transform(Kms_driven)
# MEAN = 0 and STANDARD DEVIATION = 1


years = np.array(df['Year'])
years = years.reshape(-1, 1) #converting 1-D array to 2-D array
df['Year'] = scaler.fit_transform(years)
# MEAN = 0 and STANDARD DEVIATION = 1

dependent_Y = df['Price']
independent_X = df.drop(['Price'], axis = 'columns')

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(independent_X, dependent_Y, train_size = 0.8, random_state = 1)


#MODEL BUILDING
print("LINEAR REGRESSION")
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)
Y_predicted = lr.predict(X_test)

print("The first 5 predicted values are: ")
print(Y_predicted[0:5])

print("The first 5 test values are: ")
print(np.array(Y_test[0:5]))

from sklearn.metrics import r2_score
lr_score = r2_score(Y_test, Y_predicted)
print(lr_score)

from sklearn.model_selection import cross_val_score
print(cross_val_score(lr,independent_X, dependent_Y, cv = 10))
print(np.average(cross_val_score(lr,independent_X, dependent_Y, cv = 10)), '\n')
print()
print()
### LASSO REGRESSION(prevents overfitting)

print("LASSO REGRESSION")

from sklearn.linear_model import Lasso
l = Lasso()

l.fit(X_train, Y_train)

Y_predicted_lasso = l.predict(X_test)

print("The first 5 predicted values are: ")
print(Y_predicted_lasso[0:5])

print("The first 5 test values are: ")
print(np.array(Y_test[0:5]))

l_score = r2_score(Y_test, Y_predicted_lasso)
print(l_score)

print(cross_val_score(l,independent_X, dependent_Y, cv = 10))
print(np.average(cross_val_score(l,independent_X, dependent_Y, cv = 10)), '\n')
print()
print()


###### RIDGE REGRESSION

print("RIDGE REGRESSION")

from sklearn.linear_model import Ridge

r = Ridge()

r.fit(X_train, Y_train)

Y_predicted_r = r.predict(X_test)

print("The first 5 predicted values are: ")
print(Y_predicted_r[0:5])

print("The first 5 test values are: ")
print(np.array(Y_test[0:5]))

r_score = r2_score(Y_test, Y_predicted_r)
print(r_score)

print(cross_val_score(r,independent_X, dependent_Y, cv = 10))
print(np.average(cross_val_score(r,independent_X, dependent_Y, cv = 10)), '\n')
print()
print()


###### DECISION TREE
from sklearn.tree import DecisionTreeRegressor

print("Decision Tree Regressor")
dtr = DecisionTreeRegressor()

dtr.fit(X_train, Y_train)
Y_predicted_dtr = dtr.predict(X_test)

print("The first 5 predicted values are: ")
print(Y_predicted_dtr[0:5])

print("The first 5 test values are: ")
print(np.array(Y_test[0:5]))

dtr_score = r2_score(Y_test, Y_predicted_dtr)
print(dtr_score)

print(cross_val_score(dtr,independent_X, dependent_Y, cv = 10))
print(np.average(cross_val_score(dtr,independent_X, dependent_Y, cv = 10)), '\n')
print()
print()


import pickle #way of saving a model

#saving a model
with open('CarPricePrediction.pickle', 'wb') as f:
    pickle.dump(r, f)

#loading a model
with open('CarPricePrediction.pickle', 'rb') as f:
    mp = pickle.load(f)