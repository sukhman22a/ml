# plot has some problem
''''it plots lines correctly 
but there is some problem with points(marker) 
there is a gap between years where there are no points
'''
import pandas as pd
import matplotlib.pyplot as mpl
from sklearn import linear_model
import numpy as np
df = pd.read_csv("data/canada_per_capita_income.csv")
reg = linear_model.LinearRegression() 

reg.fit(df[["year"]].values,df['per capita income (US$)'].values)
coeff = reg.coef_
incpt = reg.intercept_


analysed_df = pd.read_csv("analysed/2nd.csv")
last_year = analysed_df['year'].max()
till_now = int(input(f"enter the year above {last_year}: "))

new_prediction = []
for i in range(last_year,till_now+1):
    predicted_revenue = reg.predict([[i]])
    new_record = {"year" : i,"per capita income (US$)" : predicted_revenue}
    new_prediction.append(new_record)

new = pd.DataFrame(new_prediction)
df = pd.concat([df,new] , ignore_index=True) 
mpl.plot(df[['year']],df[["per capita income (US$)"]],marker='.')
mpl.xlabel("Year")
mpl.ylabel("Per Capita Income (US$) ")
mpl.show()
file_path = "C:/Users/LENOVO/Desktop/ml/ml/analysed/2nd.csv"
df.to_csv(file_path,index=False,na_rep="NaN")
