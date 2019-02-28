# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 13:51:30 2019

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
from sklearn import model_selection
import matplotlib.pyplot as plt


test_proportion = 0.5  # set crossval proportion
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

#%%


from sklearn.svm import SVC

degree = np.arange(1,3,step=2,dtype=int)
error = np.zeros((2,len(degree)))

for d in degree:
    svm = SVC(kernel='poly',degree=d)
    # Train the model on training data
    svm = svm.fit(X_train, y_train)
    
    y_est_test = svm.predict(X_test)
    y_est_train = svm.predict(X_train)
    
    test_class_error = 1-np.mean(y_est_test == y_test)
    train_class_error = 1-np.mean(y_est_train == y_train)
    error[0,int(d/2)], error[1,int(d/2)]= train_class_error, test_class_error

plt.plot(degree, error[0,:])
plt.plot(degree, error[1,:])
plt.xlabel('Model complexity (degree of polynomial)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over polynomial degree')
plt.show()  
errorSum = test_proportion*error[1,:] + (1-test_proportion)*error[0,:] # mean error
error_equal = sum(error)
print('Lowest average error is {:.4f} at {} using {} percent for testing'.format(min(errorSum), degree[np.argmin(errorSum)],test_proportion*100))
print('Lowest unweighted error is {:.4f} at {}'.format(min(error_equal)/2,degree[np.argmin(error_equal)]))
print('Lowest test error is {:.4f} at {}'.format(min(error[1,:]),degree[np.argmin(error[1,:])]))