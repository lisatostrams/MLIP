# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:01:24 2019

@author: Lisa
"""

import pandas as pd
from sklearn import tree
from Toolbox import treeprint
from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
data = pd.read_csv('Data/preprocessedTrain2.csv') #import data
X = data.loc[:, data.columns != 'AdoptionSpeed'] #create X without labels
X = X.fillna(0)
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)
X = X.drop('Unnamed: 0',axis=1)
X = X.drop('img_metadata_label',axis=1)
y = data['AdoptionSpeed'] #label vector

test_proportion = 0.5  # set crossval proportion
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)


#%%

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


print('Lowest test error is {:.4f} at {}'.format(min(error[1,:]), levels[np.argmin(error[1,:])]))

minl = levels[np.argmin(error[1,:])]
dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=minl) #train decision tree
dtc = dtc.fit(X_train,y_train)




#%%
attributeNames = list(X)
levels = range(1,len(attributeNames))
error = np.zeros((2,len(levels)))


attribute_importance = [(attributeNames[i],dtc.feature_importances_[i]) for i in range(len(attributeNames))]
attributes_sorted = sorted(attribute_importance, key=lambda item: item[1], reverse=True)


print('Features in order of importance:')
print(*['{}: {:.4f}'.format(i[0],i[1]) for i in attributes_sorted],sep='\n')

for t in levels:
    attributes = [i[0] for i in attributes_sorted][:t]
    dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=minl)
    Xnew = X_train[attributes]
    Xnew_test = X_test[attributes]
    dtc = dtc.fit(Xnew,y_train)
    y_est_test = dtc.predict(Xnew_test)
    y_est_train = dtc.predict(Xnew)
    
    test_class_error = 1-np.mean(y_est_test == y_test)
    train_class_error = 1-np.mean(y_est_train == y_train)
    error[0,t-1], error[1,t-1]= train_class_error, test_class_error

plt.plot(levels, error[0,:])
plt.plot(levels, error[1,:])
plt.xlabel('Model complexity (number of attributes used)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over number of attributes')
plt.show()  
errorSum = sum(error)
print('Lowest error is {:.4f} at {}'.format(min(errorSum)/2, np.argmin(errorSum)))
print('Lowest test error is {:.4f} at {}'.format(min(error[1,:]), levels[np.argmin(error[1,:])]))
print('Lowest train error is {:.4f} at {}'.format(min(error[0,:]), np.argmin(error[0,:])))

#%%

nAttributes = range(16,50)
y_pred_train = np.zeros((len(y_train),len(nAttributes)))
y_pred_test = np.zeros((len(y_test),len(nAttributes)))
error = np.zeros((2,len(nAttributes)))
adj = min(nAttributes)
for t in nAttributes:
    attributes = [i[0] for i in attributes_sorted][:t]
    dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=minl)
    Xnew = X_train[attributes]
    Xnew_test = X_test[attributes]
    dtc = dtc.fit(Xnew,y_train)
    y_pred_test[:,t-adj] = dtc.predict(Xnew_test)
    y_pred_train[:,t-adj] = dtc.predict(Xnew)
    
    test_class_error = 1-np.mean(y_pred_test[:,t-adj] == y_test)
    train_class_error = 1-np.mean(y_pred_train[:,t-adj] == y_train)
    error[0,t-adj], error[1,t-adj]= train_class_error, test_class_error

plt.plot(nAttributes, error[0,:])
plt.plot(nAttributes, error[1,:])
plt.xlabel('Model complexity (max tree depth)')
plt.ylabel('Error (misclassification rate)')
plt.legend(['Error_train','Error_test']) 
plt.title('Cross validation over tree depth')
plt.show()  
errorSum = sum(error)
print('Lowest error is {:.4f} at {}'.format(min(errorSum)/2, nAttributes[np.argmin(errorSum)]))
print('Lowest test error is {:.4f} at {}'.format(min(error[1,:]), nAttributes[np.argmin(error[1,:])]))
print('Lowest train error is {:.4f} at {}'.format(min(error[0,:]), nAttributes[np.argmin(error[0,:])]))

y_pred = pd.Series(y_test.values,index=y_test.index)
for i in range(0,len(y_pred)):
    y_pred.iloc[i] = max(Counter(y_pred_test[i,:]))