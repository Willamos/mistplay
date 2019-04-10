#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
import matplotlib.pyplot as plt
import pickle
from transformers import *


# model lib
import mlp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


# In[2]:


df = pd.read_csv('challenge.csv')
df.head(10)


# In[3]:


df.describe()


# In[4]:


df.isna().sum()


# basically i can see that the last two columns are nan almost every time

# In[5]:


df.astype(bool).sum(axis=0)


# In[6]:


df['y'].value_counts()


# In[7]:


df.head(5)


# In[8]:


df['x23'].value_counts()


# In[9]:


df1 = df.loc[df['x23'] != '0']
print(len(df1))


df1['x23'] = df1['x23'].apply(lambda x: 0 if x == 'FALSE' else 1)
df1['x23'].unique()


# In[10]:


df1['x2'].value_counts()


# In[11]:


len(df1['x2'].value_counts())


# In[12]:


df['x3'].value_counts()


# # should discard x1 and x3

# In[13]:


df1['x5'].value_counts()


# # discard x15

# In[14]:


df1['x15'].value_counts()


# In[15]:


df1['x16'].value_counts()


# In[16]:


df1['x17'].value_counts()


# In[17]:


df1['x24'].value_counts()


# In[18]:


df1['x25'].value_counts()


# # discard x24 and x25

# In[19]:


df1.loc[df['y'] > 0]['x4'].value_counts()


# In[20]:


sum(df1.loc[df1['y'] > 0]['x4'].value_counts()[:3])


# In[21]:


df1.loc[df1['y'] > 0]['x4'].value_counts().keys()[:3]


# In[22]:


sum(df1['x4'].value_counts()[:10])


# In[23]:


df1['x9'].value_counts()


# # i've decided to discard x1 (ids), x2 (android version), x3 (cel model), x6 (because x7 is the age, so i dont need the year of birth), x9 (i have no idea what it is and i cant get few columns from it), x15 (cel brand), x24, x25, x26, 'Unnamed: 27', and 'Unnamed: 28'

# In[24]:


cols_to_drop = ['x1', 'x2', 'x3' , 'x6', 'x9', 'x15'] + list(df1)[-5:]
cols_to_drop


# In[25]:


df1['x17'].unique()


# In[26]:


df['x4'].unique()


# In[27]:


numeric_cols = ['x7', 'x11', 'x12', 'x14', 'x16', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23']
cat_cols = ['x4', 'x5', 'x8', 'x10','x13', 'x17']


# In[28]:


transf = make_column_transformer((StandardScaler(), numeric_cols),
                                (OneHotEncoder(), cat_cols))


# In[29]:


df1['x4'].value_counts()


# In[30]:


df1['x5'].value_counts()


# In[31]:


df1['x8'].value_counts()


# # filtering data to get a less unbalanced dataset

# In[32]:


positives = df1.loc[df1['y'] > 0]
print(len(positives))


# # getting randomized exaples of negatives

# In[33]:


negatives_size = 1*len(positives)
negatives = df1.loc[df1['y'] == 0]
train_negatives = negatives.sample(negatives_size)
print(len(train_negatives))
train_negatives.head(3)


# In[34]:


types = df1.dtypes
types


# In[35]:


df1 = negatives.append(positives)
df1.to_csv('clean.csv', index=False)
del(df1)
del(df)
df1 = pd.read_csv('clean.csv')
positives = df1.loc[df1['y'] > 0]
negatives_size = int(1*len(positives))
negatives = df1.loc[df1['y'] == 0]
train_negatives = negatives.sample(negatives_size)
print(len(train_negatives))
train_negatives.head(3)


# In[36]:


train_negatives.describe()


# In[37]:


positives.describe()


# In[38]:


testing_data = train_negatives.append(positives)
testing_data['y'] = testing_data['y'].apply(lambda x: 1 if x>0 else 0)
len(testing_data)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split


# In[40]:


X_columns = list(df1)
X_columns.remove('y')
df1_X = df1[X_columns]


# In[41]:


pipeline = Pipeline(steps=[
                ('drop_columns', ColRemoverTransformer(cols_to_drop)),
                ('type_setter', ColDataTypeWorker(num_cols=numeric_cols, cat_cols=cat_cols)),
                ('column_transformers', transf)])
pipeline.fit(df1_X)


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(testing_data[X_columns], testing_data['y'], train_size=0.8)

X_train = pipeline.transform(X_train)
X_test = pipeline.transform(X_test)


# In[43]:


X_train.shape


# In[44]:


gs = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators': [5, 10, 15, 20, 25], 'max_depth': np.arange(10,26, 5)})


# In[45]:


gs.fit(X_train, y_train)


# In[46]:


estimator = gs.best_estimator_
preds = estimator.predict(X_test)
preds


# In[47]:


from sklearn.metrics import accuracy_score

print(accuracy_score(preds, y_test))


# In[99]:


pickle.dump(estimator, open('rf_model.mdl', 'wb'))

