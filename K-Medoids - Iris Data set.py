import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn_extra.cluster import KMedoids
import warnings # current version of seaborn generates a bunch of warnings that we'll igno
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, figure


dataset = pd.read_csv('Iris1.csv')

dataset.head()

dataset.info()

dataset.describe()

dataset.isnull().sum()


dataset.drop_duplicates(inplace=True)


# In[11]:


X = dataset.iloc[:, [1,2,3,4]].values 


# In[42]:


print(X)


# In[25]:


# Fitting K-Means to the dataset
kmed = KMedoids(n_clusters = 5)
y_kmed = kmed.fit_predict(X)
print(y_kmed)


# In[26]:


kmed.cluster_centers_


# In[27]:


kmed.inertia_


# In[28]:


kmed.n_iter_


# In[29]:


# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmed == 0, 0], X[y_kmed == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmed == 1, 0], X[y_kmed == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmed == 2, 0], X[y_kmed == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(X[y_kmed == 3, 0], X[y_kmed == 3, 1], color = 'black', label = 'Cluster 4',s=50)
sns.scatterplot(X[y_kmed == 4, 0], X[y_kmed == 4, 1], color = 'violet', label = 'Cluster 5',s=50)
sns.scatterplot(kmed.cluster_centers_[:, 0], kmed.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(True)
plt.title('IRIS Dataset')
plt.legend()
plt.show()
savefig('../results/cluster1.png')


# In[17]:


labels = kmed.labels_


# In[18]:


print(labels)


# In[19]:


kmed.labels_[:5]


# In[20]:


# Fitting K-Means to the dataset
kmed = KMedoids(n_clusters = 3)
y_kmed = kmed.fit_predict(X)
print(y_kmed)


# In[21]:


kmed.inertia_


# In[22]:


kmed.n_iter_


# In[24]:


# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmed == 0, 0], X[y_kmed == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmed == 1, 0], X[y_kmed == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmed == 2, 0], X[y_kmed == 2, 1], color = 'green', label = 'Cluster 3',s=50)

sns.scatterplot(kmed.cluster_centers_[:, 0], kmed.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(True)
plt.title('IRIS Dataset')
plt.legend()
plt.show()
savefig('../results/cluster.png')

# In[ ]:




