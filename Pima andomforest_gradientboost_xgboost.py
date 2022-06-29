#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[4]:


diabetes_new = pd.read_csv('/content/drive/MyDrive/Join Research TCE/Diabetes/pimadiabetes.csv')
diabetes_new


# In[5]:


#from google.colab import files
#uploaded = files.upload()


# In[6]:


import pandas as pd

# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)


# In[7]:


diabetes_new.head()


# In[8]:


diabetes_new["class"].value_counts()


# In[9]:


# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the diabetes features.
diabetes_new.plot(kind="scatter", x="Age", y="Glucose")


# In[10]:


diabetes_new.plot(kind="scatter", x="Age", y="BloodPressure")


# In[11]:


# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="Age", y="Glucose", data=diabetes_new, size=10)


# In[12]:


sns.lineplot(x="Age", y="Glucose", data=diabetes_new, size=10)


# In[13]:


# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(diabetes_new, hue="class", size=5)    .map(plt.scatter, "Age", "Glucose")    .add_legend()


# In[14]:


pip install category_encoders


# In[15]:


#Declare feature vector and target variable 
X = diabetes_new.drop(['class'], axis=1)

y = diabetes_new['class']


# In[16]:


#Declare feature vector and target variable 
X 


# In[17]:


y 


# In[18]:


#Split data into separate training and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[19]:


# check the shape of X_train and X_test

X_train.shape, X_test.shape


# In[20]:


#Feature Engineering 
# check data types in X_train

X_train.dtypes


# In[21]:


#Encode categorical variables
X_train.head()


# In[22]:


X_train


# In[23]:


X_test


# In[24]:


# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Age", y="Glucose", data=diabetes_new)


# In[25]:



# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier


# In[26]:


# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)


# In[27]:


# fit the model

rfc.fit(X_train, y_train)


# In[28]:


# Predict the Test set results

y_pred = rfc.predict(X_test)


# In[29]:


# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[30]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[31]:


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)


# In[32]:


# fit the model to the training set

rfc_100.fit(X_train, y_train)


# In[33]:


# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)


# In[34]:


# Check accuracy score 

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))


# In[35]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[36]:


estimator = rfc.estimators_[5]


# In[37]:


from sklearn.datasets import load_diabetes
iris = load_diabetes()
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                               rounded = True, proportion = False, 
                precision = 2, filled = True)


# In[38]:


from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


# In[39]:


## Gradient Boosting CLASSIFIER CODE STARTS HERE

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier


# In[40]:


scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[41]:


state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)


# In[42]:


lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))


# In[43]:


# LEARNING RATE = 1
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


# In[43]:





# In[44]:


# LEARNING RATE = 0.5
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


# In[45]:


## Gradient Boosting CLASSIFIER CODE ENDS HERE


# In[46]:


## Extended Gradient Boosting (XGBOOST) CLASSIFIER CODE STARTS HERE

from xgboost import XGBClassifier


# In[47]:


xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)


# In[48]:


## VALIDATION TEST SCORE
score = xgb_clf.score(X_val, y_val)
print(score)


# In[49]:


predictions=xgb_clf.predict(X_val)
print("Confusion Matrix:")
print(confusion_matrix(y_val,predictions))
print(classification_report(y_val,predictions))


# In[50]:


## Extended Gradient Boosting CLASSIFIER CODE ENDS HERE

