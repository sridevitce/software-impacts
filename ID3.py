import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
#import matplotlib.pyplot.scatter 
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, figure

import seaborn as sns
import warnings # current version of seaborn generates a bunch of warnings that we'll igno
warnings.filterwarnings("ignore")
#print(os.getcwd())
#os.chdir("C:\\Users\\Sridevi\\downloads")
Iris = pd.read_csv('Iris1.csv')
Iris.head()
sns.set(style="white", color_codes=True)
Iris["Species"].value_counts()
#scatterplot
Iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
savefig('../results/plot.png')
sns.boxplot(x="Species", y="PetalLengthCm", data=Iris)
savefig('../results/boxplot.png')
sns.FacetGrid(Iris, hue="Species", size=6) .map(sns.kdeplot, "SepalLengthCm") .add_legend()
savefig('../results/Facet.png')
sns.pairplot(Iris.drop("Id", axis=1), hue="Species", size=4)
savefig('../results/pairplot.png')

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
X, y = iris.data, iris.target
#split dataset in features and target variable
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
#feature_cols = list(iris.data.columns.values)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print("Tree Generated")
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, figure
plt.figure(figsize=(10,5))
tree.plot_tree(clf)
plt.show()
savefig('../results/graph1.png')