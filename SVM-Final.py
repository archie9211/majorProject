#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[14]:


import pandas 
import scipy 
import numpy 
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
import joblib
from sklearn.metrics import confusion_matrix


# # Read Data

# In[7]:


dataframe = pandas.read_csv("finol.csv")
dataframe.describe()
# ds = dataframe.sample(frac=1)
dataframe = dataframe.to_numpy()


# # Data Normalization

# In[8]:


X = dataframe[:,0:4]
y = dataframe[:,4]
scaler = MinMaxScaler(feature_range=(0, 1)) 
rescaledX = scaler.fit_transform(X) 
X=rescaledX


# # Split data into train and test set

# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=283)


# # Initialize SVM Model and Train

# In[10]:


model = svm.SVC(kernel='rbf', gamma = 100)

model.fit(X_train,y_train)


# # Test the model with test data

# In[11]:


acc = model.score(X_test,y_test)
print("Accuracy of model is " + str(acc*100) + "%")


# # Save Model

# In[13]:


filename = 'finalized_model.sav'
joblib.dump(model, filename)


# # COnfusion matrix for test set

# In[15]:


y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

