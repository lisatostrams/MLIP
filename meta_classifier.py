# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:39:15 2019

@author: Lisa
"""

#%% Attributes obtained from crossvalidation
nAttributes = ['popularity resquer id', 'Age', 'Breed1', 'PhotoAmt', 'description length', 'sentiment', 'img_ave_contrast',
               'State', 'Sterilized', 'img_metadata_sentiment2', 'MaturitySize', 'Color1', 'FurLength', 'Breed2',
               'Quantity', 'Fee', 'Gender', 'spayed', 'cute', 'Name', 'Dewormed', 'ador', 'abandon']
max_depth = 7

n_estimators = 10

tol = 0.01

nn = 15

xgb_params = {
    'eval_metric': 'rmse',
    'seed': 1337,
    'verbosity': 0,
}


#%%
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('Data/preprocessedTrain2.csv') #import data
X = data.loc[:, data.columns != 'AdoptionSpeed'] #create X without labels
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)

y = data['AdoptionSpeed'] #label vector

test_proportion = 0.5  # set crossval proportion
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=test_proportion)

X_train = X_train[nAttributes]
X_test = X_test[nAttributes]

# For log reg

Xlr_train = X_train
Xlr_test = X_test
dummy = ['State','Type','Breed1','Breed2','Gender','Color1','Color2','Color3','Vaccinated','Dewormed','Sterilized']
for d in dummy:
    if(d in nAttributes):
        one_hottr = pd.get_dummies(Xlr_train[d],prefix=d)
        # Drop column d as it is now encoded
        Xlr_train = Xlr_train.drop(d,axis = 1)
        # Join the encoded df
        Xlr_train = Xlr_train.join(one_hottr)
        
        one_hot = pd.get_dummies(Xlr_test[d],prefix=d)

        for col in list(one_hottr):
            if(col not in list(one_hot)):
                one_hot[col]=0
        for col in list(one_hot):
            if(col not in list(one_hottr)):
                one_hot = one_hot.drop(col,axis=1)
        Xlr_test = Xlr_test.drop(d,axis = 1)
        Xlr_test = Xlr_test.join(one_hot)

#%% train all classifiers
        
classifiers = 'DTC RF LOGREG KNN SVM GNB XGB'.split(sep=' ')
predictions = np.zeros((len(y_test),len(classifiers)))

dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=max_depth) #train decision tree
dtc = dtc.fit(X_train,y_train)
predictions[:,0] = dtc.predict(X_test)

rf = RandomForestRegressor(n_estimators = n_estimators)
rf = rf.fit(X_train, y_train)
predictions[:,1] = np.round(rf.predict(X_test),0)

logreg = LogisticRegression(tol=tol,)
logreg = logreg.fit(Xlr_train, y_train)
predictions[:,2] = logreg.predict(Xlr_test)

knn = KNeighborsClassifier(nn)
knn = knn.fit(X_train,y_train)
predictions[:,3] = knn.predict(X_test)

svm = SVC(tol=tol)
svm = svm.fit(X_train, y_train)
predictions[:,4] = svm.predict(X_test)

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
predictions[:,5] = gnb.predict(X_test)

d_train = xgb.DMatrix(data=X_train, label=y_train, feature_names=X_train.columns)
d_test = xgb.DMatrix(data=X_test, label=y_test, feature_names=X_test.columns)
evallist = [(d_test, 'eval'), (d_train, 'train')]
model = xgb.train(dtrain=d_train, num_boost_round=30000, evals=evallist, early_stopping_rounds=5000,  params=xgb_params)
predictions[:,6] = np.round(model.predict(d_test, ntree_limit=model.best_ntree_limit),0)

#%%
correct = np.zeros((len(y_test),len(classifiers)))
for i in range(len(classifiers)):
    err = 1-np.mean(predictions[:,i]==y_test)
    print('Test error for {} is: {:.4f}'.format(classifiers[i],err))
    correct[:,i] = predictions[:,i] == y_test
    if(min(predictions[:,i])<0):
        print(classifiers[i])
        predictions[predictions[:,i]<0,i] = 0
    if(np.any(np.isnan(predictions[:,i]))):
        print(classifiers[i])
        
    if(max(predictions[:,i]>4)):
        print(classifiers[i])
        predictions[predictions[:,i]>4,i] = 4
    if(np.all(np.isfinite(predictions[:,i]))==0):
        print(classifiers[i])
        
correctdf = pd.DataFrame(correct)
print('In total, {:.2f}% of the testset is classified correctly by at least one classifier'.format(correctdf.max(axis=1).mean()*100))


#%%
from sklearn.neural_network import MLPClassifier
import seaborn as sn
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm,y):
    df_cm = pd.DataFrame(cm, index = [i+1 for i in np.unique(y)],
                  columns = [i+1 for i in np.unique(y)])
    plt.figure()
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted class')
    plt.ylabel('Actual class')
    plt.show()
    
    
from sklearn import model_selection
cv = model_selection.KFold(n_splits=10)
train_error = np.zeros(10)
test_error = np.zeros(10)

k=0
for train_index,test_index in cv.split(predictions):
    mlp_train = predictions[train_index,:]
    mlpy_train = y_test.iloc[train_index]
    mlp_test = predictions[test_index,:]
    mlpy_test = y_test.iloc[test_index]
    print('Fold {}'.format(k))
   
    best_train = np.inf
    best_test = np.inf
    for i in range(5):
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10,))
        clf.fit(mlp_train, mlpy_train) 
        c = 1- clf.score(mlp_train,mlpy_train)
        if(c < best_train):
            best_train=c
        e_test = 1- clf.score(mlp_test,mlpy_test)
        if(e_test < best_test):
            best_test=e_test
    train_error[k] = best_train
    test_error[k] = best_test
    k+=1

print('Average training error: {} \nAverage test error: {}'.format(np.mean(train_error),np.mean(test_error)))


#test = pd.read_csv('Data/preprocessedtest.csv')
#X_test = test.drop('Description',axis=1) #drop non numerical values
#id = X_test['PetID']
#X_test = X_test.drop('PetID',axis=1) #
#X_test = X_test.drop('RescuerID',axis=1)
#X_test = X_test[attributes]
#y_est = dtc.predict(X_test)