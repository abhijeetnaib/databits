#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 14:23:59 2018

@author: abhijeet
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('/home/abhijeet/Desktop/ML/logisticv_regression/day.csv')
df.info()
df.shape

def mapping(x):
    if (x <= 2000):
        return(1)
    elif(x <=4000):
        return(2)
    elif x <=6000:
        return(3)
    elif x <=8000:
        return(4)
    else:
        return(5)
        

df[df['cnt']<=2000].count()

#df['cnt'].mean()

df['DEMAND'] = df['cnt'].apply(lambda x: mapping(x))

#df1.info()
df1['DEMAND'].value_counts()

df1.head(1)

#df1=df1.drop(['casual','registered','dteday','cnt'],axis=1)

df1.info()

df1.groupby('season').mean()
'''
sns.distplot(df1['cnt'],hist=False, rug=True);
sns.jointplot(x="instant", y="cnt", data=df1);
sns.jointplot(x="instant", y="cnt", data=df1,kind="kde");
'''

df2
df2=df.columns.values.tolist()

y=['DEMAND']

X=[i for i in df2 if i not in y ]

X

df1.info()

df3=df.drop(['DEMAND','dteday'],axis=1)

Y=df['DEMAND']
Y

df3.info()

X_train,X_test,Y_train,Y_test=train_test_split(df3,Y,test_size=0.25)

X_train.shape
Y_train.shape
X_test.shape
Y_test.shape

logreg = LogisticRegression(multi_class='multinomial',solver='newton-cg')
'''
rfe = RFE(logreg, 9)
rfe = rfe.fit(X_train, Y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
'''
logreg.fit(X_train,Y_train)

import seaborn as sns
sns.boxplot(x=df['cnt'])


from scipy import stats
import numpy as np

z = np.abs(stats.zscore(df['cnt']))
print(np.where(z > 3))

dir(logreg)



y_pred = logreg.predict(X_test)



print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test)))

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_test,y_pred))

print(confusion_matrix(Y_test,y_pred))



'''
from sklearn import tree
clf = tree.DecisionTreeClassifier()
dtree = clf.fit(X_train,Y_train)
dres=clf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(Y_test,dres))

print(confusion_matrix(Y_test,dres))
'''

