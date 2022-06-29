#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install -c conda-forge scikit-learn-extra


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids


# In[3]:


print(os.getcwd())


# In[4]:


os.chdir("C:\\Users\\Sridevi\\downloads")


# In[5]:


dataset = pd.read_csv('cars.csv')


# In[6]:


dataset.head()


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


dataset.isnull().sum()


# In[10]:


dataset.drop_duplicates(inplace=True)


# In[11]:


dataset.isna().sum()


# In[11]:


X = dataset.iloc[:, [1,2,3,4]].values 


# In[13]:


dataset.mean()


# In[14]:


dataset.median()


# In[15]:


dataset.std()


# In[16]:


dataset.var()


# In[23]:


x=dataset['income']
y=x=dataset['sales']
plt.scatter(x,y)


# In[26]:


data=dataset['miles']
fig=plt.figure(figsize=(8,6))
plt.boxplot(data)
plt.show()


# In[27]:


from sklearn import preprocessing


# In[31]:


d2=dataset[['income','sales']]
preprocessing.normalize(d2)


# In[ ]:




