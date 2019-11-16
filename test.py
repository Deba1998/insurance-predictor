# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 12:35:38 2019

@author: DELL
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

#categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])
labelencoder_X_3 = LabelEncoder()
X[:, 5] = labelencoder_X_3.fit_transform(X[:, 5])
onehotencoder = OneHotEncoder(categorical_features = [5])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)
#measuring RMSE
from math import sqrt
from sklearn.metrics import mean_squared_error 
result=sqrt(mean_squared_error(y_test,y_pred))

