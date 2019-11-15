# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:40:57 2019

@author: DELL
"""

import keras 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# Importing the dataset
dataset = pd.read_csv('insurance.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values
# Encoding categorical values
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
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)



#model fitting
model = keras.Sequential()
model.add(keras.layers.Dense(output_dim=8,input_dim=8))
model.add(keras.layers.Dense(output_dim=8))
model.add(keras.layers.Dense(output_dim=1))
model.compile(optimizer='adam',loss='mean_squared_error')

model.fit(X, y, epochs=150,batch_size=10)

y_pred=model.predict(X, batch_size=10)
from math import sqrt
from sklearn.metrics import mean_squared_error 
result=sqrt(mean_squared_error(y,y_pred))
print(sqrt(36645121.9006))