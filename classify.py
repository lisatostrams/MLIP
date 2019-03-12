# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 12:49:15 2019

@author: Lisa
"""
#%% Attributes obtained from crossvalidation
nAttributes = ['Age',
 'popularity resquer id',
 'Breed1',
 'img_pixels',
 'PhotoAmt',
 'Sterilized',
 'description length',
 'img_ave_contrast',
 'Breed2',
 'Quantity',
 'Gender',
 'img_metadata_sentiment2',
 'beaut',
 'MaturitySize',
 'State',
 'Color3',
 'vaccin',
 'abandon',
 'Vaccinated',
 'Fee',
 'indoor',
 'cute',
 'great']
max_depth = 8

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
from sklearn.preprocessing import PowerTransformer
from sklearn import preprocessing



data = pd.read_csv('Data/preprocessedTrain3.csv') #import data
X = data.loc[:, data.columns != 'AdoptionSpeed'] #create X without labels
X = X.fillna(0)
X = X.drop('Description',axis=1) #drop non numerical values
X = X.drop('PetID',axis=1) #
X = X.drop('RescuerID',axis=1)
X = X.drop('Unnamed: 0',axis=1)
X = X.drop('Unnamed: 0.1',axis=1)
X = X.drop('img_metadata_label',axis=1)
X = X[nAttributes]
y = data['AdoptionSpeed'] #label vector

test = pd.read_csv('Data/preprocessedtest3.csv')


X_test = test.drop('Description',axis=1) #drop non numerical values
X_test = X_test.fillna(0)
id = X_test['PetID']
X_test = X_test.drop('PetID',axis=1) #
X_test = X_test.drop('RescuerID',axis=1)
X_test = X_test.drop('Unnamed: 0',axis=1)
X_test = X_test.drop('Unnamed: 0.1',axis=1)

X_test = X_test.drop('img_metadata_label',axis=1)
X_test = X_test[nAttributes]


non_zer0 = np.mean(X==0)==0
zero = non_zer0[non_zer0.values==False].index
non_zer0 = non_zer0[non_zer0.values==True].index

scaler = preprocessing.PowerTransformer(method='box-cox', standardize=True).fit(X[non_zer0])
X[non_zer0] = scaler.transform(X[non_zer0])
X_test[non_zer0] = scaler.transform(X_test[non_zer0])
scaler = preprocessing.StandardScaler().fit(X[zero])
X[zero] = scaler.transform(X[zero])
X_test[zero] = scaler.transform(X_test[zero])

meta_train, meta_test, meta_y_train, meta_y_test = model_selection.train_test_split(X,y,test_size=0.1,stratify=y)
Xlr_train = meta_train
Xlr_m_test = meta_test
Xlr_test = X_test
dummy = ['State','Type','Breed1','Breed2','Gender','Color1','Color2','Color3','Vaccinated','Dewormed','Sterilized']
for d in dummy:
    if(d in nAttributes):
        
        train = pd.get_dummies(Xlr_train[d],prefix=d)
        test = pd.get_dummies(Xlr_test[d],prefix=d)
        m_test = pd.get_dummies(Xlr_m_test[d],prefix=d)
        result = set(list(train))
        result.intersection_update(list(test))
        result.intersection_update(list(m_test))
        one_hottr = train[list(result)]
        one_hot = test[list(result)]
        one_hotm = m_test[list(result)]
        Xlr_train = Xlr_train.drop(d,axis = 1)
        # Join the encoded df
        Xlr_train = Xlr_train.join(one_hottr)
        
        Xlr_test = Xlr_test.drop(d,axis = 1)
        Xlr_test = Xlr_test.join(one_hot)
        Xlr_m_test = Xlr_m_test.drop(d,axis=1)
        Xlr_m_test = Xlr_m_test.join(one_hotm)




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

logreg = LogisticRegression(tol=tol,solver='liblinear',multi_class='auto')
logreg = logreg.fit(Xlr_train, meta_y_train)
predictions[:,2] = logreg.predict(Xlr_test)
mlp_train[:,2] = logreg.predict(Xlr_train)
mlp_test[:,2] = logreg.predict(Xlr_m_test)

knn = KNeighborsClassifier(nn)
knn = knn.fit(meta_train, meta_y_train)
predictions[:,3] = knn.predict(X_test)
mlp_train[:,3] = knn.predict(meta_train)
mlp_test[:,3] = knn.predict(meta_test)

svm = SVC(tol=tol,gamma='auto')
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
model = xgb.train(dtrain=d_train, num_boost_round=30000, evals=evallist, early_stopping_rounds=3000, verbose_eval=3000, params=xgb_params)
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
    print()
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
def hist(y):
    histy = np.zeros((5,))
    for i in range(0,5):
        histy[i] = np.mean(y==i)
    return histy

def sse_hist(h1,h2,weight=[1,1,1,1,1]):
    sse = 0
    for i in range(0,5):
        sse = sse + weight[i]*(h1[i] - h2[i])**2
    return sse
#%%
histy = hist(y)     

from sklearn.neural_network import MLPClassifier
models = []
accuracy = []
models_sse = []
sse = []
for i in range(1,10):
    model_j = []
    score_j = []
    sse_j = []
    for j in range(0,10):
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(i,))
        clf.fit(mlp_test, meta_y_test)
        model_j.append(clf)
        score_j.append(clf.score(mlp_test,meta_y_test))
        hist_est = hist(clf.predict(predictions))
        weights = 1-histy
        sse_j.append(sse_hist(hist_est,histy,weight=weights))
        
    
    print("Layer {} test accuracy: {:.4f}".format(i,max(score_j)))
    print("Layer {} lowest SSE hist: {:.4f}".format(i,min(sse_j)))
    print()
    
    models.append(model_j[np.argmax(score_j)])
    accuracy.append(max(score_j))
    sse.append(sse_j[np.argmax(score_j)])
    
    
#%%
model_acc = np.argmax(accuracy)
print('Best number of hlayers test acc = {}'.format(model_acc+1)) 
model_sse = np.argmin(sse)
print('Best number of hlayers hist sse = {}'.format(model_sse+1)) 
s = [accuracy[i]-sse[i] for i in range(0,7)]
model = np.argmax(s)
print('Best number of hlayers total = {}'.format(model+1)) 
#%%
clf = models[model_acc]
print(clf.score(mlp_train,meta_y_train)) 
print(clf.score(mlp_test,meta_y_test))

y_est = clf.predict(predictions)
#%%


submission = pd.DataFrame(index=X_test.index)
submission['PetID'] = id
submission['AdoptionSpeed'] = y_est
submission['AdoptionSpeed'] = submission['AdoptionSpeed'].astype(int)
submission.to_csv('submission.csv',index=False)

histy = y.hist()
submission.hist()
plt.show()

#%%

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
fig, ax = plt.subplots()
objects = 'DTC RF LOGREG KNN SVM GNB XGB'.split(sep=' ')

performance = [np.mean(mlp_train[:,i] == meta_y_train) for i in range(len(objects))]
performance2 = [np.mean(mlp_test[:,i] == meta_y_test) for i in range(len(objects))]
performance.append(clf.score(mlp_train,meta_y_train))
performance2.append(clf.score(mlp_test,meta_y_test))
objects.append('Meta classifier MLP')
index = np.arange(len(objects))

importance = [(objects[i],performance[i],performance2[i]) for i in range(len(objects))]
isorted = sorted(importance, key=lambda item: item[2], reverse=True)

objects = [i[0] for i in isorted]
performance = [i[1] for i in isorted]
performance2 = [i[2] for i in isorted]

bar_width = 0.35

rects1 = ax.barh(index, performance, bar_width,align='center', alpha=0.5,label='Meta train data')
rects2 = ax.barh(index+bar_width, performance2,bar_width, align='center', alpha=0.5,label='Meta test data')
plt.yticks(index+bar_width-0.175, objects)

for i, v in enumerate(performance2):
    ax.text(v + 0.01, i + .25, '{:.4f}'.format(v), color='darkorange',fontweight='bold')
ax.legend()
plt.title('Accuracy of the classifiers')
plt.tight_layout()
plt.savefig('classifiers.png',dpi=300)
plt.show()

#%%

