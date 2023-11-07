#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# In[1]:


def save_report(exp_name, df, report_or_cm = "classification_report"):  
    exps_dir = "exps"
    if not os.path.exists(exps_dir):
        os.makedirs(exps_dir)
    exp_dir = os.path.join(exps_dir, exp_name) 
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    saved_file_path = os.path.join(exp_dir, exp_name+"_"+report_or_cm+".csv") 
    df.to_csv(saved_file_path)


# In[7]:


def scale_data(df, cols): 
    rob_scaler = RobustScaler()
    i = 0
    for col in cols:
        df.insert(i,f'scaled_{col}', rob_scaler.fit_transform(df[col].values.reshape(-1,1)))
        i += 1  
    df.drop(cols, axis=1, inplace=True)


# In[9]:


def split_data(df, test_size):
    X = df.iloc[:, :-1].values #input features as numpy array
    X_df =  df.iloc[:, :-1] #input features as dataframe
    print(X.shape) 

    y = df.iloc[:,-1:].values #output target as numpy array
    y_df = df.iloc[:,-1:] #output target as dataframe
    print(y.shape) 

    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y)  
    # startify ----> to be sure that the fraud ratio is equally distributed in both train and test sets 

    train_unique_label, train_counts_label = np.unique(y_train, return_counts=True)
    test_unique_label, test_counts_label = np.unique(y_test, return_counts=True)

    print('\n Train set')
    print (train_unique_label)
    print (train_counts_label)
    print((train_counts_label * 100) /len(y_train), '%')

    print('\n Test set')
    print (test_unique_label)
    print (test_counts_label)
    print((test_counts_label * 100) /len(y_test), '%')
    
    return X_train, X_test, y_train, y_test


# In[ ]:




