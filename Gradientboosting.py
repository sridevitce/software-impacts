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

## Gradient Boosting CLASSIFIER CODE STARTS HERE

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
#Split data into separate training and test set 
from sklearn.model_selection import train_test_split
#Declare feature vector and target variable 
X = diabetes_new.drop(['class'], axis=1)

y = diabetes_new['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  test_size=test_size, random_state=state)
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=20, learning_rate=learning_rate, max_features=2, max_depth=2, random_state=0)
    gb_clf.fit(X_train, y_train)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(X_val, y_val)))
    # LEARNING RATE = 1
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=1, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


# LEARNING RATE = 0.5
gb_clf2 = GradientBoostingClassifier(n_estimators=20, learning_rate=0.5, max_features=2, max_depth=2, random_state=0)
gb_clf2.fit(X_train, y_train)
predictions = gb_clf2.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))


