#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


print(os.getcwd())


# In[3]:


os.chdir("C:\\Users\\Sridevi\\downloads")


# In[34]:





# In[35]:


dataset.head()


# In[36]:


dataset.info()


# In[37]:


dataset.describe()


# In[38]:


dataset.isnull().sum()


# In[9]:


dataset.drop_duplicates(inplace=True)


# In[39]:


X = dataset.iloc[:, [1,2,3,4]].values 


# In[40]:


print(X)


# In[41]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
    


# In[42]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)    


# In[39]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 10), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()


# In[43]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)


# In[44]:


# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(True)
plt.title('IRIS Dataset')
plt.legend()
plt.show()


# In[21]:


labels = kmeans.labels_
print(labels)


# In[45]:


kmeans.inertia_


# In[46]:


kmeans.n_iter_


# In[47]:


kmeans.cluster_centers_


# In[25]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)


# In[26]:


kmeans.cluster_centers_


# In[27]:


kmeans.inertia_


# In[28]:


kmeans.n_iter_


# In[29]:


# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'black', label = 'Cluster 4',s=50)
sns.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'violet', label = 'Cluster 5',s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(True)
plt.title('IRIS Dataset')
plt.legend()
plt.show()


# In[31]:


labels = kmeans.labels_


# In[32]:


print(labels)


# In[48]:


kmeans.labels_[:5]


# In[ ]:




