#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 19:03:18 2018

@author: abhijeet
"""

from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df=pd.read_csv('/home/abhijeet/Desktop/ML/logisticv_regression/advertising.csv')


df.head(2)

df.info()
df.describe()

sns.jointplot(df['Age'],df['Daily Time Spent on Site'])
sns.jointplot(df['Area Income'],df['Daily Time Spent on Site'])
sns.jointplot(df['Area Income'],df['Age'])

sns.jointplot(df['Area Income'],df['Daily Internet Usage'])
sns.jointplot(df['Age'],df['Daily Internet Usage'])
sns.jointplot(df['Daily Time Spent on Site'],df['Daily Internet Usage'])

sns.pairplot(df,hue='Clicked on Ad')


df_x =df[['Age','Area Income','Daily Internet Usage','Daily Time Spent on Site','Male',]]
df_y=df['Clicked on Ad']

X_train,X_test,Y_train,Y_test=train_test_split(df_x,df_y,test_size=0.3)


logreg = LogisticRegression()

logreg.fit(X_train,Y_train)

predictions=logreg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_test,predictions))

print(confusion_matrix(Y_test,predictions))
