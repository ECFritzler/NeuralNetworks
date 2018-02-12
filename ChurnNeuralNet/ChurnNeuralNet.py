# -*- coding: utf-8 -*-
""" Artificial Neural Network for Churn Modelling Data

This program requires the Keras Library
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values # Independent Vars
y = dataset.iloc[:, 13].values # Dependent vars

#Encoding categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encodeCountry = LabelEncoder()
x[:, 1] = encodeCountry.fit_transform(x[:, 1])
encodeGender = LabelEncoder()
x[:, 2] = encodeGender.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:] # Getting rid of the first column to get rid of the dummy var trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, 
                                                test_size = 0.2, 
                                                random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

