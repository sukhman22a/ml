"""Cross  model validation """
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

iris = load_iris()
# print(dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
lr = cross_val_score(LogisticRegression(max_iter= 200),x,y)
print(lr)
svc = cross_val_score(SVC(),x,y)
print(svc)
dt = cross_val_score(DecisionTreeClassifier(criterion="entropy"),x,y)
print(dt)
rfc = cross_val_score(RandomForestClassifier(),x,y)
print(rfc)