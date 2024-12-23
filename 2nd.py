"""train and test """

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
df = pd.read_csv("data/carprices (bmw).csv")
x = df[["Mileage","Age(yrs)"]]
y = df[["Sell Price($)"]]

x_train ,x_test ,y_train ,y_test = train_test_split(x,y, test_size=0.2 , random_state=10)
# random-state is there so values dont get randomised as we rerun the program multiple times
fig, ax = plt.subplots()
ax.set_xlabel("Sell price($)")
ax.set_ylabel("mileage")
ax.plot(df["Sell Price($)"],df["Mileage"],"ro")

ax2 = ax.twinx()

ax2.set_ylabel("age(years)")
ax2.plot(df["Sell Price($)"],df["Age(yrs)"],"yo")
fig.tight_layout()#it delete the space b/w axis
plt.show()

# from graph both mileage and age show linear regression wrt sell price .
# so linear_regression_model is applicable.

model = LinearRegression()
model.fit(x_train.values,y_train.values)
print(model.predict(x_test.values))
print(y_test)
print(model.score(x_test,y_test))