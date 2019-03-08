# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:49:15 2019

@author: Lisa
"""
#%% Attributes obtained from crossvalidation
nAttributes = ['popularity resquer id', 'Age', 'Breed1', 'PhotoAmt', 'description length', 'sentiment', 'img_ave_contrast',
               'State', 'Sterilized', 'img_metadata_sentiment2', 'MaturitySize', 'Color1', 'FurLength', 'Breed2',
               'Quantity', 'Fee', 'Gender', 'cute', 'Name', 'Dewormed', 'ador', 'abandon']
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
X = X.fillna(0)
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)
X = X.drop('Unnamed: 0',axis=1)
X = X.drop('img_metadata_label',axis=1)
X = X[nAttributes]
y = data['AdoptionSpeed'] #label vector

test = pd.read_csv('Data/preprocessedtest2.csv')

X_test = test.drop('Description',axis=1) #drop non numerical values
X_test = X_test.fillna(0)
id = X_test['PetID']
X_test = X_test.drop('PetID',axis=1) #
X_test = X_test.drop('RescuerID',axis=1)
X_test = X_test.drop('Unnamed: 0',axis=1)
X_test = X_test.drop('img_metadata_label',axis=1)
X_test = X_test[nAttributes]

meta_train, meta_test, meta_y_train, meta_y_test = model_selection.train_test_split(X,y,test_size=0.5)
Xlr_train = X

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


meta_train, meta_test, meta_y_train, meta_y_test = model_selection.train_test_split(X,y,test_size=0.5)
Xlr_train, metaXlr_test, ylr_train, ylr_test = model_selection.train_test_split(Xlr_train,y,test_size=0.5)
#%%


classifiers = 'DTC RF LOGREG KNN SVM GNB XGB'.split(sep=' ')
predictions = np.zeros((len(X_test),len(classifiers)))
mlp_train = np.zeros((len(meta_train),len(classifiers)))
mlp_test = np.zeros((len(meta_test),len(classifiers)))

dtc = tree.DecisionTreeClassifier(criterion='gini',max_depth=max_depth) #train decision tree
dtc = dtc.fit(meta_train,meta_y_train)
predictions[:,0] = dtc.predict(X_test)
mlp_train[:,0] = dtc.predict(meta_train)
mlp_test[:,0] = dtc.predict(meta_test)

rf = RandomForestRegressor(n_estimators = n_estimators)
rf = rf.fit(meta_train, meta_y_train)
predictions[:,1] = np.round(rf.predict(X_test),0)
mlp_train[:,1] = np.round(rf.predict(meta_train),0)
mlp_test[:,1] = np.round(rf.predict(meta_test),0)

logreg = LogisticRegression(tol=tol,)
logreg = logreg.fit(Xlr_train, ylr_train)
predictions[:,2] = logreg.predict(Xlr_test)
mlp_train[:,2] = logreg.predict(Xlr_train)
mlp_test[:,0] = logreg.predict(metaXlr_test)

knn = KNeighborsClassifier(nn)
knn = knn.fit(meta_train, meta_y_train)
predictions[:,3] = knn.predict(X_test)
mlp_train[:,3] = knn.predict(meta_train)
mlp_test[:,3] = knn.predict(meta_test)

svm = SVC(tol=tol)
svm = svm.fit(meta_train, meta_y_train)
predictions[:,4] = svm.predict(X_test)
mlp_train[:,4] = svm.predict(meta_train)
mlp_test[:,4] = svm.predict(meta_test)

gnb = GaussianNB()
gnb = gnb.fit(meta_train, meta_y_train)
predictions[:,5] = gnb.predict(X_test)
mlp_train[:,5] = dtc.predict(meta_train)
mlp_test[:,5] = knn.predict(meta_test)



d_train = xgb.DMatrix(data=meta_train, label=meta_y_train, feature_names=meta_train.columns)
d_val = xgb.DMatrix(data=meta_test,label=meta_y_test, feature_names=meta_test.columns)
evallist = [(d_val, 'eval'), (d_train, 'train')]
model = xgb.train(dtrain=d_train, num_boost_round=30000, evals=evallist, early_stopping_rounds=3000,  params=xgb_params)
predictions[:,6] = np.round(model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns), ntree_limit=model.best_ntree_limit),0)


mlp_train[:,6] = np.round(model.predict(xgb.DMatrix(meta_train, feature_names=meta_train.columns), ntree_limit=model.best_ntree_limit),0)
mlp_test[:,6] = np.round(model.predict(xgb.DMatrix(meta_test, feature_names=meta_test.columns), ntree_limit=model.best_ntree_limit),0)

#%%

correct = np.zeros((len(meta_train),len(classifiers)))
correct_test = np.zeros((len(meta_test),len(classifiers)))
for i in range(len(classifiers)):
    err = 1-np.mean(mlp_train[:,i]==meta_y_train)
    print('Train error for {} is: {:.4f}'.format(classifiers[i],err))
    err = 1-np.mean(mlp_test[:,i]==meta_y_test)
    print('Test error for {} is: {:.4f}'.format(classifiers[i],err))
    correct[:,i] = mlp_train[:,i] == meta_y_train
    correct_test[:,i] = mlp_test[:,i] == meta_y_test
    if(min(mlp_train[:,i])<0):
        print(classifiers[i])
        mlp_train[mlp_train[:,i]<0,i] = 0
    if(np.any(np.isnan(mlp_train[:,i]))):
        print(classifiers[i])
        
    if(max(mlp_train[:,i]>4)):
        print(classifiers[i])
        mlp_train[mlp_train[:,i]>4,i] = 4
    if(np.all(np.isfinite(mlp_train[:,i]))==0):
        print(classifiers[i])
        
correctdf = pd.DataFrame(correct)
correct_testdf = pd.DataFrame(correct_test)
print('In total, {:.2f}% of the meta training set is classified correctly by at least one classifier'.format(correctdf.max(axis=1).mean()*100))
print('In total, {:.2f}% of the meta test set is classified correctly by at least one classifier'.format(correct_testdf.max(axis=1).mean()*100))

#%%

from sklearn.neural_network import MLPClassifier
models = []
errors= []
for i in range(1,15):
    model_j = []
    error_j = []
    for j in range(0,10):
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(i,))
        clf.fit(mlp_train, meta_y_train)
        model_j.append(clf)
        error_j.append(clf.score(mlp_test,meta_y_test))
    
    print("Layer {}".format(i))

    print(max(error_j))
    
    models.append(model_j[np.argmax(error_j)-1])
    errors.append(max(error_j))
    
#%%
model = np.argmax(error_j)
print('Best number of hlayers = {}'.format(model)) 
#%%
clf = models[model]
print(clf.score(mlp_train,meta_y_train)) 
print(clf.score(mlp_test,meta_y_test))

y_est = clf.predict(predictions)
#%%


submission = pd.DataFrame(index=X_test.index)
submission['PetID'] = id
submission['AdoptionSpeed'] = y_est
submission.to_csv('submission.csv',index=False)
