# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 12:43:32 2019

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

#Xlr = X.drop('State',axis=1)

dummy = ['State','Type','Breed1','Breed2','Gender','Color1','Color2','Color3','Vaccinated','Dewormed','Sterilized']
for d in dummy:
    one_hot = pd.get_dummies(Xlr[d],prefix=d)
    # Drop column d as it is now encoded
    Xlr = Xlr.drop(d,axis = 1)
    # Join the encoded df
    Xlr = Xlr.join(one_hot)

test_proportion = 0.5  # set crossval proportion
Xlr_train, Xlr_test, ylr_train, ylr_test = model_selection.train_test_split(Xlr,y,test_size=test_proportion)

#%%
from sklearn.linear_model import LogisticRegression

tol = [1,0.1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]
error = np.zeros((2,len(tol)))
for t in tol:
    logreg = LogisticRegression(tol=t,)
    logreg.fit(Xlr_train, ylr_train)
    y_est_test = logreg.predict(Xlr_test)
    y_est_train = logreg.predict(Xlr_train)
    
    test_class_error = 1-np.mean(y_est_test == ylr_test)
    train_class_error = 1-np.mean(y_est_train == ylr_train)
    error[0,int(-np.log10(t))], error[1,int(-np.log10(t))]= train_class_error, test_class_error

plt.semilogx(tol, error[0,:])
plt.semilogx(tol, error[1,:])
plt.xlabel('Convergence (tolerance)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over number of trees in forest')
plt.show()  


errorSum = test_proportion*error[1,:] + (1-test_proportion)*error[0,:] # mean error
error_equal = sum(error)
print('Lowest average error is {:.4f} at {} using {} percent for testing'.format(min(errorSum), tol[np.argmin(errorSum)],test_proportion*100))
print('Lowest unweighted error is {:.4f} at {}'.format(min(error_equal)/2,tol[np.argmin(error_equal)]))
print('Lowest test error is {:.4f} at {}'.format(min(error[1,:]),tol[np.argmin(error[1,:])]))
