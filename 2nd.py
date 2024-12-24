import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as mpl 

df = pd.read_csv("data/insurance_data.csv")
mpl.scatter(df['age'].values,df["bought_insurance"].values,marker="+",color ="yellow")
mpl.xlabel("age")
mpl.ylabel("bought Insurance")
mpl.show()
 
x = df[["age"]]
# [[]] to get in 2d array because it is needed in .fit() parameter
# if removed one [] then till .fit( ) part code will run smoothly but at .fit() it will require mltidimensional array

y=df["bought_insurance"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=0)

model = LogisticRegression()
model.fit(x_train,y_train.values)
print(model.predict(x_test))
print(model.score(x_test,y_test))