# -*- coding: utf-8 -*-
""" Artificial Neural Network for Churn Modelling Data

This program requires the Keras Library
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:13].values # Independent Vars
y = dataset.iloc[:, 13].values # Dependent vars

#Encoding categorical Data
encodeCountry = LabelEncoder()
x[:, 1] = encodeCountry.fit_transform(x[:, 1])
encodeGender = LabelEncoder()
x[:, 2] = encodeGender.fit_transform(x[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:,1:] # Getting rid of the first column to get rid of the dummy var trap

# Splitting the dataset into the Training set and Test set
xTrain, xTest, yTrain, yTest = train_test_split(x, y, 
                                                test_size = 0.2, 
                                                random_state = 0)

# Feature Scaling
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

# Initializing the Neural Network
classifier = Sequential()

# Adding Input and Fisrt Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', 
                     activation = 'relu', input_dim = 11))

# Adding Second Hidden Layer
classifier.add(Dense(output_dim = 6, init = 'uniform', 
                     activation = 'relu'))

# Adding Output Layer
classifier.add(Dense(output_dim = 1, init = 'uniform', 
                     activation = 'sigmoid'))

# Compiling the Neural Network - Applying Stochastic Gradient Descent
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# Fitting Neural Net to Training Data
classifier.fit(xTrain, yTrain, batch_size = 10, nb_epoch = 100)

# Predicting Test Results
yPred = classifier.predict(xTest)
yPred = (yPred > 0.5)
# Create the Confusion Matrix
matrix = confusion_matrix(yTest, yPred)