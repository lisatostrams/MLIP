# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 08:45:56 2019

@author: Lisa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Data/preprocessedTrain.csv') #import data
X = data.loc[:, data.columns != 'AdoptionSpeed'] #create X without labels
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)

y = data['AdoptionSpeed'] #label vector

attributeNames = list(X.columns.values) #for printing purposes
classNames = ['sameDay','firstWeek','firstMonth','2nd3rdMonth','notAdopted']

#%%

from sklearn import tree
from Toolbox import treeprint
from sklearn import model_selection
import matplotlib.pyplot as plt

test_proportion = 0.5  # set crossval proportion
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)
levels = range(2,51)
error = np.zeros((2,len(levels)))

for t in levels:
    dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=t) #train decision tree
    dtc = dtc.fit(X_train,y_train)
    
    y_est_test = dtc.predict(X_test)
    y_est_train = dtc.predict(X_train)
    
    test_class_error = 1-np.mean(y_est_test == y_test)
    train_class_error = 1-np.mean(y_est_train == y_train)
    error[0,t-2], error[1,t-2]= train_class_error, test_class_error

plt.plot(levels, error[0,:])
plt.plot(levels, error[1,:])
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over tree depth')
plt.show()  
errorSum = sum(error)
print('Lowest error is {:.2f} at {}'.format(min(errorSum)/2, np.argmin(errorSum)))

#%%

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 10 decision trees

estimators = np.logspace(1,3,num=3,base=10)
error = np.zeros((2,len(estimators)))

for n_e in estimators:
    rf = RandomForestRegressor(n_estimators = int(n_e))
    # Train the model on training data
    rf = rf.fit(X_train, y_train)
    
    y_est_test = rf.predict(X_test)
    y_est_train = rf.predict(X_train)
    
    test_class_error = 1-np.mean(y_est_test == y_test)
    train_class_error = 1-np.mean(y_est_train == y_train)
    error[0,int(np.log10(n_e)-1)], error[1,int(np.log10(n_e)-1)]= train_class_error, test_class_error

#%%
plt.semilogx(estimators, error[0,:])
plt.semilogx(estimators, error[1,:])
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over tree depth')
plt.show()  
errorSum = sum(error)
print('Lowest error is {:.2f} at {}'.format(min(errorSum)/2, np.argmin(errorSum)))


#%%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


#%%

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


#%%

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)


#%%


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)
