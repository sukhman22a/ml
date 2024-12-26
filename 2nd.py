from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
iris = load_iris()
# print(dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
X_train, X_test, y_train, y_test = train_test_split(iris.data,iris.target,test_size=0.2)
model = LogisticRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))