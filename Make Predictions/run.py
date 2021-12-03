#!/usr/bin/env python
# coding: utf-8

# In[116]:


import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


# In[58]:


rf = joblib.load('./random_forest.joblib')
knn = joblib.load('./KNN.joblib')
svm = joblib.load('./SVM.joblib')
dTree = joblib.load('./DecisionTree.joblib')
lm = joblib.load('./Logistic_Regression.joblib')
ann = tf.keras.models.load_model('ANN_Model.h5')
models = {'Logistic Regression':lm,'SVM':svm,'KNN':knn,'Decision Tree':dTree,'Random Forest': rf,'ANN':ann}
labels = {0:'building windows (float processed)',1: 'building windows (non-float processed)',
          2: 'vehicle windows (float processed)',3: 'containers',4: 'tableware',5: 'headlamps'}


# In[139]:


while True:
    s = input("Provide glass data(separated by comma)[type `f frac` to score on a sample. q to quit]\n")
    if s[0].lower() =='q':
        print('Goodbye...')
        break
    elif s[0].lower() == 'f':
        l = s.split(' ')
        frac = float(l[1])
        df = pd.read_csv('glass.data',header=None)
        df = df.iloc[:,1:]
        oh = OneHotEncoder()
        oh.fit(np.array(df.iloc[:,-1]).reshape(-1,1))
        df_sample = df.sample(frac=frac)
        X,y = df_sample.iloc[:,:-1],df_sample.iloc[:,-1]
        X_test = StandardScaler().fit_transform(X)
        for name,model in models.items():
            y_test = oh.transform(np.array(y).reshape(-1,1)) if name=='ANN' else LabelEncoder().fit_transform(y) 
            pred = model.predict(X_test)
            if name=='ANN':
                pred = np.argmax(pred,axis=1)
                y_test = np.argmax(y_test,axis=1)
            acc = accuracy_score(pred,y_test)
            print(f"Accuracy for {name}: {round(acc,3)}")
    else:
        l = s.split(',')
        x = []
        for i in l:
            x.append(float(i))
        x_s = StandardScaler().fit_transform(np.array(x).reshape(-1,9))
        print()
        for name,model in models.items():
            pred =np.argmax(model.predict(x_s)[0]) if name=='ANN' else model.predict(x_s)[0]
            print(f"{name}: {labels[pred]}")
    print()

