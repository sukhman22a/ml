# Dummy variables using pandas
# dummy trap ?
# Theroy is important for ml

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/carprices.csv")
ohe = OneHotEncoder()

dfle = df
dfle["Car Model"] = ohe.fit_transform(dfle[["Car Model"]]).values
