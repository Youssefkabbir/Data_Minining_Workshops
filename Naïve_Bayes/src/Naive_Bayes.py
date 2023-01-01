# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 17:04:00 2023

@author: 21260
"""
# import the Libraries
import numpy as nm 
import matplotlib.pyplot as mtp  
import pandas as pd 
import os.path
print('I  have All  Libraries imported')

# importing the Datasets
dataset=pd.read_csv(os.path.join(os.path.dirname(__file__), "resources\Data\Social_Network_Ads.csv"))
X=dataset.iloc[:,[0,1]].values
Y=dataset.iloc[:,2]
#spliting the Dataset to Training and test sets 
from sklearn.model_selection  import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)
#Fiting the Naive Bayes to Training set 
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, Y_train)
#Prediction of the test Result
Y_pred=classifier.predict(X_test)
#Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

# Visualising the Training set results  
from matplotlib.colors import ListedColormap  
x_set, y_set = X_train, Y_train  
X1, X2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(X1, X2, classifier.predict(nm.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
             alpha = 0.75, cmap = ListedColormap(('purple', 'green')))  
mtp.xlim(X1.min(), X1.max())  
mtp.ylim(X2.min(), X2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Naive Bayes (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  