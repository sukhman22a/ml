from sklearn.model_selection import train_test_split
from sklearn import tree 
import pandas as pd
df = pd.read_csv("data/titanic.csv")
# print(df)
required_df = df[["Survived","Pclass","Sex","Age","Fare"]]
x = required_df.drop(["Survived"],axis="columns")
y = required_df["Survived"]
dummies = pd.get_dummies(required_df["Sex"]).astype(int)
x = pd.concat([x,dummies],axis="columns")
x = x.drop(["Sex","male"],axis="columns")
x["Age"] = x['Age'].fillna(x["Age"].median())
x["Fare"] = x['Fare'].fillna(x["Fare"].mean()) 

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model = tree.DecisionTreeClassifier()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))