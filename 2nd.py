# Dummy variables using pandas
# dummy trap ?
# Theroy is important for ml

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data/carprices.csv")
dummies = pd.get_dummies(df["Car Model"]).astype(int)
df_updated = pd.concat([df,dummies],axis="columns")
final = df_updated.drop(["Car Model","Audi A5"], axis="columns")

x = final.drop(["Sell Price($)"],axis="columns")
y = final["Sell Price($)"]

model = LinearRegression()
model.fit(x.values,y.values)

pre = model.predict([[70000,3,0,0]])
# mileage , age(yrs), BMW X5(0),mercedez benz c class(0),audi a5(1)

print(model.predict([[45000,4,0,1]]))
print(model.predict([[86000,7,1,0]]))
print(model.score(x.values,y.values))
# x.values give the values in numoy array and x alone give dataframe
