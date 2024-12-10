import pandas as pd
import matplotlib.pyplot as mpl
from sklearn import linear_model
import numpy as np
df = pd.read_csv("data\homeprice.csv")
reg = linear_model.LinearRegression() 

reg.fit(df[["area"]].values,df.price)
# .values returns only the data from the DataFrame, not the column names.
# If you don't use it properly, it may cause a warning."
print(reg.predict([[3030]]))


print("hello")

