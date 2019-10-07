#!/usr/bin/env python
# coding: utf-8

# # Fraud Detection 
# **Categorical Features - Transaction**
# 
# * ProductCD
# * card1 - card6
# * addr1, addr2
# * P_emaildomain
# * R_emaildomain
# * M1 - M9
# 
# 
# **Categorical Features - Identity**
# * DeviceType
# * DeviceInfo
# * id_12 - id_38

# Create list of all categorical features

# In[1]:


t_c = ['ProductCD'] +       ['card%i' %(i) for i in range(1,7)] +       ['addr1','addr2','P_emaildomain','R_emaildomain'] +       ['M%i' %(i) for i in range(1,9)]

i_c = ['DeviceType', 'DeviceInfo'] +       ['id_%i' %(i) for i in range(12,38)]

c = t_c + i_c


# In[2]:


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', 999) 


# In[3]:


import os
from os.path import join
wd = os.path.abspath('')


# In[4]:


get_ipython().run_cell_magic('time', '', "train_identity = pd.read_csv(join(wd,'../data/train_identity.csv'))\ntrain_transaction = pd.read_csv(join(wd,'../data/train_transaction.csv'))\n\ntest_identity = pd.read_csv(join(wd,'../data/test_identity.csv'))\ntest_transaction = pd.read_csv(join(wd,'../data/test_transaction.csv'))\nprint('Files Loaded!!')")


# ## Indeity files 
# The indeity files have a number of variables.
# 
# We have a key *TransactionID*
# 
# This file appears to be the idenity of the device that was used to make the transaction

# In[5]:


print('\t'.join(train_identity.columns.to_list()))


# In[6]:


train_identity[i_c].iloc[1]


# ## Transaction files
# This file is where our labeled data is.
# 
# * TransactionID = Key
# * isFraud = Label
# * TransactionDT = DateTime??
# * TransactionAmt = Transaction Amount
# * ProductCD = ?         
# * card1             
# * card2             
# * card3             
# * card4             
# * card5             
# * card6             
# * addr1             
# * addr2             
# * dist1             
# * dist2             
# * P_emaildomain     
# * R_emaildomain     
# 

# In[7]:


print('\t'.join(train_transaction.columns.to_list()))


# In[8]:


train_transaction[t_c].iloc[0]


# # Missing Variables
# ## Train
# ### Transaction

# In[9]:


def how_much_missing(df_missing, c):    
    missing_values_count = df_missing.isnull().sum()
    print (missing_values_count[c])
    total_cells = np.product(df_missing.shape)
    total_missing = missing_values_count.sum()
    print ("% of missing data = ",(total_missing/total_cells) * 100)


# In[10]:


how_much_missing(train_transaction, t_c)


# Thats alot of missing data

# In[11]:


how_much_missing(train_identity, i_c)


# ## Test
# ### Transaction

# In[12]:


how_much_missing(test_transaction, t_c)


# In[13]:


how_much_missing(test_identity, i_c)


# # Balance of fruad vs Non-fraud
# 
# Data is very biased towards non-fruad events

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
train_transaction.isFraud.value_counts().plot.bar()
_ = plt.xlabel('isFraud')


# # Date Time

# In[16]:


train_max_dt = train_transaction['TransactionDT'].max()
train_min_dt = train_transaction['TransactionDT'].min()

print("Training data datetime min: %i max: %i" % (train_min_dt, train_max_dt))

test_max_dt = test_transaction['TransactionDT'].max()
test_min_dt = test_transaction['TransactionDT'].min()

print("Test data datetime min: %i max: %i" % (test_min_dt, test_max_dt))


# To look at later:
# https://www.kaggle.com/jesucristo/fraud-complete-eda/notebook#Time-vs-fe
# 
# We are not told what the time unit is on TransactionnDT. Link above shows these are seconds. See below:
# 
# 

# In[17]:


import math
train_transaction['day'] = train_transaction['TransactionDT'].apply(lambda x: math.floor((x - train_min_dt) / (60*60*24)))

train_transaction['hour'] = train_transaction['TransactionDT'].apply(lambda x: math.floor((x - train_min_dt) / (60*60)))


# In[18]:


_ = train_transaction.groupby(['hour']).count()['TransactionID'].head(24*7).plot()


# The above shows a daily period

# # Joining Transaction with Identity

# We can join the tgwo files together using the *TranactionID*.
# 
# Joining on the left to ensure all training data is kept

# In[22]:


train_transaction.set_index('TransactionID')

df = train_transaction.join(train_identity.set_index('TransactionID') 
                     ,on = 'TransactionID'
                     ,how = 'left' )


# How many rows could not be joined?

# In[23]:


not_joined = df[['id_%02i' % i for i in range(1,39)]].                isnull().                all(1).                sum()

print('%i were not joined out of %i' % (not_joined, df.shape[0]))


# So the majority could not be joined :/
# 
# 
# Just as a sainty check lets double check that there were no all null from the start

# In[24]:


train_identity[['id_%02i' % i for i in range(1,39)]].    isnull().    all(1).    sum()


# In[25]:


df[df['isFraud'] == 1]


# In[26]:


df = df.set_index('TransactionID')


# In[27]:


df[c]


# In[28]:


df['ProductCD'].value_counts()


# In[29]:


df['card4'].value_counts()


# In[30]:


df['card6'].value_counts()


# In[31]:


df['P_emaildomain'].value_counts()


# In[32]:


df['R_emaildomain'].value_counts()


# In[33]:


df['DeviceType'].value_counts()


# # The Mmmms

# In[34]:


for i in range(1,10):  
    print(df['M%i' % i].value_counts())


# mostly true false, but m3 seems to refer to previous Ms

# In[35]:


df[['M%i' % i for i in range(1,10)]].sample(10, random_state = 123)


# No clear patteren, i thought m4 might tell you which is not null

# ### OneHot encoding

# In[36]:


to_encode = ['ProductCD','card4','card6','P_emaildomain','R_emaildomain' , 'DeviceType','DeviceInfo'] + ['M%i' % i for i in range(1,10)] 


dfs = [pd.get_dummies(df[i], prefix = i ) for i in to_encode]


# In[37]:


from functools import reduce
features = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), dfs)


# In[38]:


# clean column names
features.columns = [i.replace(' ','_').replace('.', '_') for i in features.columns.to_list()]


# ## The Cs

# In[39]:


df['C1'].plot()


# ## Add conintous variables

# In[40]:


features = features.join(df['TransactionAmt'])


# In[41]:


features = features.join(df['isFraud'])


# In[42]:




# # Train Model

# ## Split Data

# In[43]:


from sklearn import linear_model
clf = linear_model.SGDClassifier(max_iter=1000, 
                                 tol=1e-3, 
                                 random_state = 1234,
                                 verbose = 1)


# In[90]:


clf.fit(X_train, Y_train)


# # Evaluate Model

# In[92]:





# In[93]:





# In[48]:


def scores(clf, Y_test, X_test):
    from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
    y_true = Y_test
    y_pred = clf.predict(X_test)
    print('Confusuion matrix ' + str(confusion_matrix(y_true = y_true, y_pred = y_pred)))
    print('Accurary of %0.2f' % accuracy_score(y_true = y_true, y_pred = y_pred))
    print('Recall of %0.2f' % recall_score(y_true = y_true, y_pred = y_pred))
    print('Precision of %0.2f' % precision_score(y_true = y_true, y_pred = y_pred))


# In[96]:





# # Random Forest

# In[44]:


from sklearn.ensemble import RandomForestClassifier


# In[51]:


clf = RandomForestClassifier(n_estimators=1000, 
                             random_state=0,
                             n_jobs = -1,
                             verbose = 1)
clf.fit(X_train, Y_train)


# In[ ]:


scores(clf, Y_test, X_test)


# In[ ]:




