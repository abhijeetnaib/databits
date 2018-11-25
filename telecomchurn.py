#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 23:34:34 2018

@author: abhijeet
"""

WA_Fn-UseC_-Telco-Customer-Churn.csv

from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df=pd.read_csv('/home/abhijeet/Desktop/ML/logisticv_regression/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.info()

cleanup_names ={
        "gender" : {"Female" : 0 , "Male" : 1},
        "Partner" :{ "Yes" : 1, "No" : 0},
        "Dependents" : {"Yes" : 1 , "No" : 0},
        "PhoneService" : {"Yes": 1, "No" : 0},
        "MultipleLines" : {"Yes": 1, "No" : 0, "No phone service":0},
        "InternetService" : {"DSL":1, "No": 0 ,"Fiber optic" : 2},
        "OnlineSecurity" :{"Yes": 1, "No" : 0, "No internet service":0},
        "OnlineBackup" :{"Yes": 1, "No" : 0, "No internet service":0},
        "DeviceProtection" :{"Yes": 1, "No" : 0, "No internet service":0},
        "TechSupport" :{"Yes": 1, "No" : 0, "No internet service":0},
        "StreamingTV" :{"Yes": 1, "No" : 0, "No internet service":0},
        "StreamingMovies" :{"Yes": 1, "No" : 0, "No internet service":0},
        "Contract" :{"Month-to-month": 1, "One year" : 0, "Two year":2},
        "PaperlessBilling" : {"Yes": 1, "No" : 0},
        "PaymentMethod" : {"Electronic check": 1, "Mailed check" : 0,
                           "Bank transfer (automatic)": 2, 
                           "Credit card (automatic)":3},
        "Churn" :{ "Yes" : 1, "No" : 0}
                           }

df['TotalCharges'].unique()


df1=df.replace(cleanup_names).dropna()



df_x=df1.drop(['Churn','TotalCharges','customerID'],axis=1)

df_y=df1['Churn']

X_train,X_test,Y_train,Y_test=train_test_split(df_x,df_y,test_size=0.3)

X_train.count()
Y_test.count()


logreg=LogisticRegression()

logreg.fit(X_train,Y_train)

predictions=logreg.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_test,predictions))

print(confusion_matrix(Y_test,predictions))


from sklearn import tree
clf = tree.DecisionTreeClassifier()
dtree = clf.fit(X_train,Y_train)

dprob=dtree.predict_proba(X_test)
dres=dtree.predict(X_test)


print(classification_report(Y_test,dres))

print(confusion_matrix(Y_test,dres))
