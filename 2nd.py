import pandas as pd
import matplotlib.pyplot as mpl
from sklearn import linear_model
import numpy as np
df = pd.read_csv("data\homeprice.csv")
reg = linear_model.LinearRegression() 

reg.fit(df[["area"]].values,df.price)

test_df = pd.read_csv("data\homeprice_test.csv")
test_df["prices"] = reg.predict(test_df[["area"]].values)
coeff = reg.coef_
incpt = reg.intercept_
test_df["checkPrice"] = coeff*test_df[["area"]] + incpt
print(test_df)
print("hello")

