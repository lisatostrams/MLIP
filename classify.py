# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:49:15 2019

@author: Lisa
"""
import pandas as pd
from sklearn import tree
from Toolbox import treeprint
from sklearn import model_selection
import matplotlib.pyplot as plt

data = pd.read_csv('Data/preprocessedTrain.csv') #import data
X = data.loc[:, data.columns != 'AdoptionSpeed'] #create X without labels
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)

y = data['AdoptionSpeed'] #label vector

dtc = tree.DecisionTreeClassifier(criterion='gini', max_depth=10) #train decision tree
dtc = dtc.fit(X,y)


#%%
attributeNames = list(X)
attribute_importance = [(attributeNames[i],dtc.feature_importances_[i]) for i in range(len(attributeNames))]
attributes_sorted = sorted(attribute_importance, key=lambda item: item[1], reverse=True)

print('Features in order of importance:')
print(*['{}: {:.4f}'.format(i[0],i[1]) for i in attributes_sorted],sep='\n')
attributes = [i[0] for i in attributes_sorted][:15]

Xnew = X[attributes]

#%%

test = pd.read_csv('Data/preprocessedtest.csv')
X_test = test.drop('Description',axis=1) #drop non numerical values
id = X_test['PetID']
X_test = X_test.drop('PetID',axis=1) #
X_test = X_test.drop('RescuerID',axis=1)

y_est = dtc.predict(X_test)

#%%

submission = pd.DataFrame(index=X_test.index)
submission['PetID'] = id
submission['AdoptionSpeed'] = y_est
submission.to_csv('submission.csv',index=False)
