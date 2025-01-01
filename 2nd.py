import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


df = pd.read_csv("data/titanic.csv")
data_set = df[["Pclass","Survived","Sex","Age","Fare"]]
target = data_set["Survived"]
features = data_set.drop(['Survived'],axis="columns")
features["female"] = df["Sex"].apply(lambda x : 1 if x == "female" else 0)
features["Age"] = features["Age"].fillna(features["Age"].mean()).astype(int)
features = features.drop(["Sex"],axis="columns")

x_train,x_test ,y_train, y_test = train_test_split(features,target,test_size=0.33)
nb = GaussianNB()
nb.fit(x_train,y_train)

print(cross_val_score(nb,features,target,cv=10).mean())