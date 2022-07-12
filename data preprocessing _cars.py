import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
#print(os.getcwd())
#os.chdir("C:\\Users\\Sridevi\\downloads")
dataset = pd.read_csv('cars.csv')
dataset.head()
dataset.info()
dataset.describe()


# In[9]:


dataset.isnull().sum()


# In[10]:


dataset.drop_duplicates(inplace=True)


# In[11]:


dataset.isna().sum()


# In[12]:


X = dataset.iloc[:, [1,2,3,4]].values 


# In[13]:


dataset.mean()


# In[14]:


dataset.median()


# In[15]:


dataset.std()


# In[16]:


dataset.var()


# In[17]:


x=dataset['income']
y=dataset['sales']
plt.scatter(x,y)


# In[18]:


data=dataset['miles']
fig=plt.figure(figsize=(8,6))
plt.boxplot(data)
plt.show()


# In[19]:


from sklearn import preprocessing


# In[20]:


d2=dataset[['income','sales']]
preprocessing.normalize(d2)


# In[24]:


array = dataset.values
print(array)


# In[27]:


# separate array into input and output components
# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
X = array[:,0:5]
Y = array[:,5]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)


# In[29]:


#summarize transformed data
numpy.set_printoptions(precision=3)
print("Values after scaling")
print(rescaledX[0:5,:])


# In[ ]:




