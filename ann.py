# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:13:16 2019

@author: Rajput
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values#input atrix
y = dataset.iloc[:, 13].values#outut matrix

#encoding categorial data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])#for countries
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])#for gender
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#buildoing ANN
import keras
from keras.models import Sequential#to initialize nn
from keras.layers import Dense#to add layer in ANN

#initilising the nn
clf=Sequential()

#adding the ip and hidden layer to network
#1st hidden layer
clf.add(Dense(units=6,activation='relu',kernel_initializer='uniform',input_shape=(11,)))
#2nd hidden layer
clf.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))
#output layer
clf.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

#compiling the nn
clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fittinhg nn to training set
clf.fit(X_train,y_train,batch_size=1,epochs=100)

#making predictions and evalluating the model

# Predicting the Test set results
y_pred = clf.predict(X_test)
y_pred= (y_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
