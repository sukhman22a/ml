'''Support Vector Machine 
for dividing two datasets
https://youtu.be/FB5EdxAGxQg?feature=shared'''


from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.svm import SVC

digits= load_digits()
# 'DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names'
df  = pd.DataFrame(digits.data, columns=digits.feature_names)
y = digits.target
x = df
model = SVC(C=6,kernel="linear",gamma="scale")
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 ,random_state=1)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))