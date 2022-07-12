import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll igno
warnings.filterwarnings("ignore")
diabetes_new = pd.read_csv('pimadiabetes.csv')
diabetes_new
#from google.colab import files
#uploaded = files.upload()
# We'll also import seaborn, a Python graphing library
sns.set(style="white", color_codes=True)
diabetes_new.head()
diabetes_new["class"].value_counts()
# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the diabetes features.
import matplotlib
matplotlib.use('Agg')
plt.figure(figsize=(10,5))
from matplotlib.pyplot import plot, savefig, figure
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
diabetes_new.plot(kind="scatter", x="Age", y="Glucose")
plt.show()
savefig('../results/scatter.png')

# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
plt.figure(figsize=(10,5))
sns.jointplot(x="Age", y="Glucose", data=diabetes_new, size=10)
plt.show()
savefig('../results/jointplot.png')

plt.figure(figsize=(10,5))
sns.lineplot(x="Age", y="Glucose", data=diabetes_new, size=10)
plt.show()
savefig('../results/lineplot.png')

# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
plt.figure(figsize=(10,5))
sns.FacetGrid(diabetes_new, hue="class", size=5).map(plt.scatter, "Age", "Glucose")
plt.show()
savefig('../results/FacetGrid.png')


#Declare feature vector and target variable 
X = diabetes_new.drop(['class'], axis=1)

y = diabetes_new['class']
print("Feature Vectors")
print(X)
print("Target Variales")
print(y)
#Split data into separate training and test set 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
# check the shape of X_train and X_test

X_train.shape, X_test.shape
#Feature Engineering 
# check data types in X_train

X_train.dtypes
#Encode categorical variables
X_train.head()
print("TRAINING SET")
print(X_train)
print("Testing Set")
print(X_test)

# We can look at an individual feature in Seaborn through a boxplot
plt.figure(figsize=(10,5))
sns.boxplot(x="Age", y="Glucose", data=diabetes_new)
plt.show()
savefig('../results/boxplot.png')


# import Random Forest classifier

from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 

rfc = RandomForestClassifier(random_state=0)
# fit the model

rfc.fit(X_train, y_train)

# Predict the Test set results

y_pred = rfc.predict(X_test)
# Check accuracy score 

from sklearn.metrics import accuracy_score

print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# instantiate the classifier with n_estimators = 100

rfc_100 = RandomForestClassifier(n_estimators=100, random_state=0)
# fit the model to the training set

rfc_100.fit(X_train, y_train)
# Check accuracy score 

# Predict on the test set results

y_pred_100 = rfc_100.predict(X_test)

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred_100)))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))



