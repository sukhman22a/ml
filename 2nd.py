import pandas as pd
import matplotlib.pyplot as mpl
from sklearn import linear_model
import numpy as np
df = pd.read_csv("data\homeprice.csv")
# print(df)
mpl.plot(df.area,df.price,color = "blue",marker= ".")
mpl.xlabel("Area")
mpl.ylabel("price")
mpl.show()
print("hello")

# it plots x and y axis values . 