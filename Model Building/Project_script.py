#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,StratifiedKFold,RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,SMOTENC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('glass.data',header=None)
df.columns = ['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']
df = df.iloc[:,1:]
print(df.head())


# In[10]:


print(df.describe())


# ###### Class 1: building windows (float processed)
# ###### Class 2: building windows (non-float processed)
# ###### Class 3: vehicle windows (float processed)
# ###### Class 4: vehicle windows (non-float processed) (0 samples in this data)
# ###### Class 5: containers
# ###### Class 6: tableware
# ###### Class 7: headlamps

# In[8]:


class_info = df.groupby('Class').count().iloc[:,0]
print(class_info)


# In[31]:


plt.hist(df['Class'])
plt.show()


# #### The dataset can be divided into
# Window Glass: 163 samples
# 
# non-window glass: 51 samples
# #### Or
# Float Glass: 87 samples
# 
# non-float glass: 76 samples
# #### which is more balanced

# In[34]:


for i in class_info.index:
    c = class_info[i]
    print(f'Class {i}, Count:{c}, Percentage:{round(c*100/214,3)}%')


# In[42]:


df.hist(figsize=(10,10))
plt.show()


# In[838]:


from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values
X_train,X_test, y_train,y_test = train_test_split(X,Y,test_size=0.33,stratify=Y,random_state=1)


# In[839]:


ss=StandardScaler()
le=LabelEncoder()
X = ss.fit_transform(X_train)
y = le.fit_transform(y_train)
X_test = ss.transform(X_test)
y_test = le.transform(y_test)
print("X,y shape:",X.shape,y.shape)


# In[840]:


def evaluate(X,y,estimator):
    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=10)
    score = cross_val_score(estimator,X,y,scoring='accuracy',cv=cv)
    return score


# ### Baseline: 35% (Predicting class 2 for every sample)

# ### We'll now evaluate several models 

# In[841]:


lm = LogisticRegression(penalty='l2',solver='newton-cg')
svm = SVC(gamma='auto')
knn = KNeighborsClassifier()
dTree = DecisionTreeClassifier(max_depth=100)
weights = {0:1.0,1:1.0,2:2.0,3:2.0,4:2.0,5:2.0}
rf = RandomForestClassifier(n_estimators=1000,class_weight=weights)

models = {'Logistic Regression':lm,'SVM':svm,'KNN':knn,'Decision Tree':dTree,'Random Forest': rf}

results = []
print("Cross val average accuracy(without oversampling or hyperparameter tuning)")
for k,v in models.items():
    result = evaluate(X,y,v)
    results.append(result)
    print(f'{k}: {round(np.mean(result),3)}')


# In[842]:


plt.boxplot(results,labels=models.keys(),showmeans=True)
plt.show()


# #### Using SMOTE(Synthetic Minority Oversampling technique) Oversampling
# #### Let's take a look again on what the class ditribution looks like

# In[843]:


plt.bar(np.unique(y),class_info)
plt.show()


# #### Using SMOTE

# In[844]:


X,y = SMOTE(sampling_strategy = {0:200, 1:200, 2:200, 3:200, 4:200, 5:200}).fit_resample(X,y)


# In[845]:


names,counts = np.unique(y,return_counts=True)
plt.bar(names,counts)
plt.show()


# #### Re-evaluate model on oversampled data

# In[846]:


lm = LogisticRegression(penalty='l2',solver='newton-cg')
svm = SVC(gamma='auto')
knn = KNeighborsClassifier()
dTree = DecisionTreeClassifier(max_depth=100)
rf = RandomForestClassifier(n_estimators=1000,class_weight='balanced')

models = {'Logistic Regression':lm,'SVM':svm,'KNN':knn,'Decision Tree':dTree,'Random Forest': rf}

results = []
print("Cross val average accuracy on oversampled data(without hyperparameter tuning)")
for k,v in models.items():
    result = evaluate(X,y,v)
    results.append(result)
    print(f'{k}: {round(np.mean(result),3)}')


# In[847]:


plt.boxplot(results,labels=models.keys(),showmeans=True)
plt.show()


# ### Hyperparameter Tuning

# In[ ]:


print("Hyperparameter Tuning")


# In[848]:


params = {'penalty':('l1', 'l2'),'tol':[1e-4,1e-3],'C':[1.0,3.0,5.0],
             'class_weight':('balanced',None),'solver':('newton-cg', 'lbfgs', 'liblinear'),'multi_class':('auto', 'ovr', 'multinomial')}
clf = GridSearchCV(LogisticRegression(),params,cv=5,scoring='accuracy')
clf.fit(X,y)
lm_best = clf.best_estimator_
print("Logistic Regresion:")
print('Best Result:',clf.best_score_)
print('Best Param:',clf.best_params_)


# In[849]:


from sklearn.model_selection import GridSearchCV
params = {'n_neighbors':[1,2,3,4,5,6,7,8,9,10],'weights':('uniform','distance'),
             'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),
         'p':[1,2,3,4]}
clf = GridSearchCV(KNeighborsClassifier(),params,cv=5,scoring='accuracy')
clf.fit(X,y)
knn_best = clf.best_estimator_
print("KNN")
print('Best Result:',clf.best_score_)
print('Best Param:',clf.best_params_)


# In[850]:


params = {'C':[1.0,1.5,2.0,2.5,3.0,4.0,5.0],'kernel':('linear', 'poly', 'rbf', 'sigmoid'),
             'gamma':('scale','auto'),'decision_function_shape':('ovo', 'ovr')}
clf = GridSearchCV(SVC(),params,cv=5,scoring='accuracy')
clf.fit(X,y)
svm_best = clf.best_estimator_
print("SVM")
print('Best Result:',clf.best_score_)
print('Best Param:',clf.best_params_)


# In[851]:


params = {'criterion':('gini', 'entropy'),'splitter':('best', 'random'),'max_depth':[10,50,100,500,1000],
          'min_samples_split':[1,2,5,10],'min_samples_leaf':[1,2,3,5],'max_features':('auto','sqrt','log2',None)}
clf = GridSearchCV(DecisionTreeClassifier(),params,cv=5,scoring='accuracy')
clf.fit(X,y)
dTree_best = clf.best_estimator_
print("Decision Tree")
print('Best Result:',clf.best_score_)
print('Best Param:',clf.best_params_)


# In[852]:


params = {'n_estimators':[1,10,100,1000],'max_depth':[1,10,100,1000],
          'min_samples_split':[1,2],'min_samples_leaf':[1,2],
         'class_weight':('balanced', 'balanced_subsample',None)}
clf = GridSearchCV(RandomForestClassifier(),params,cv=5,scoring='accuracy')
clf.fit(X,y)
rf_best = clf.best_estimator_
print("Random Forest")
print('Best Result:',clf.best_score_)
print('Best Param:',clf.best_params_)


# In[853]:


models = {'Logistic Regression':lm_best,'SVM':svm_best,'KNN':knn_best,'Decision Tree':dTree_best,'Random Forest': rf_best}

results = []
print("Cross val average accuracy on oversampled data and hyperparameter tuned model")
for k,v in models.items():
    result = evaluate(X,y,v)
    results.append(result)
    print(f'{k}: {round(np.mean(result),3)}')


# In[854]:


plt.boxplot(results,labels=models.keys(),showmeans=True)
plt.show()


# In[855]:


models = {'Logistic Regression':lm_best,'SVM':svm_best,'KNN':knn_best,'Decision Tree':dTree_best,'Random Forest': rf_best}
results = []
print("Test set accuracy")
for k,v in models.items():
    v.fit(X,y)
    y_pred = v.predict(X_test)
    results.append(accuracy_score(y_test,y_pred))
    print(f'{k}: {round(accuracy_score(y_test,y_pred),3)}')


# In[856]:


plt.bar(models.keys(),results)
plt.show()


# ### Neural Networks

# In[ ]:


print("Neural Network Training")


# In[857]:


Y=y
YY = np.zeros((len(Y) , 1))
YY[:,0] = Y
enc = OneHotEncoder(sparse = False , handle_unknown='error')
enc.fit(YY)
Y_transformed = enc.transform(YY)


# In[892]:


from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras import Sequential
def Glass_Deep_Model(dropout_rate = 0.5 , activation='relu' , optimizer='adam',loss='binary_crossentropy'):
    # create model
    model = Sequential()
    model.add(Dense(9, input_dim=9, activation=activation))
    #model.add(Dense(1024, activation=activation))
    #model.add(Dropout(dropout_rate))
    model.add(Dense(512,  activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256,  activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6,  activation="softmax"))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


# In[893]:


from keras.wrappers.scikit_learn import KerasClassifier
Final_Model = KerasClassifier(build_fn=Glass_Deep_Model, verbose=0)


# In[894]:


history = Final_Model.fit(X, Y_transformed,validation_split = 0.33, epochs=100, batch_size=16, verbose=1)


# In[906]:


plt.plot(history.history['accuracy'],label='Training')
plt.plot(history.history['val_accuracy'],label='Validation')
plt.title('Accuracy Across Epochs')
plt.legend()
plt.show()


# In[907]:


plt.plot(history.history['loss'],label='Training')
plt.plot(history.history['val_loss'],label='Validation')
plt.title('Loss Across Epochs')
plt.legend()
plt.show()


# In[898]:


preds=Final_Model.predict(X_test)
print("Confusion Matrix:\n",confusion_matrix(y_test,preds))


# In[918]:


print("Test Accuracy:",accuracy_score(preds,y_test))


# In[917]:


all_models = list(models.keys())
all_models.append('ANN')
all_results = list(results)
all_results.append(accuracy_score(preds,y_test))
print("Overall test accuracy")
for i,m in enumerate(all_models):
    print(f"{m}: {round(all_results[i],4)}")
plt.bar(all_models,all_results, align='center',width=0.7)
plt.xticks(rotation=45)
plt.show()

