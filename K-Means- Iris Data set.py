#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll igno
warnings.filterwarnings("ignore")
#print(os.getcwd())
#os.chdir("C:\\Users\\Sridevi\\downloads")
dataset = pd.read_csv('Iris1.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.isnull().sum()
dataset.drop_duplicates(inplace=True)
X = dataset.iloc[:, [1,2,3,4]].values 
print(X)
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, figure


wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)    

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 10), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.show()
savefig('../results/graph1.png')

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)
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
savefig('../results/cluster.png')

labels = kmeans.labels_
print(labels)
kmeans.inertia_
kmeans.n_iter_

kmeans.cluster_centers_

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
savefig('../results/cluster1.png')

# In[31]:


labels = kmeans.labels_

print(labels)

kmeans.labels_[:5]
