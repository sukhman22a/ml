import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import matplotlib.pyplot as mpl 

df = pd.read_csv("data/HR_comma_sep.csv")
# satisfaction_level        0.38
# last_evaluation           0.53
# number_project               2
# average_montly_hours       157
# time_spend_company           3
# Work_accident                0
# left                         1
# promotion_last_5years        0
# Department               sales
# salary                     low
# Name



# drop_df = df.drop(["salary","Department"],axis="columns")
# val = drop_df.groupby("left").mean()
# print(val)
# # # from val 
# # # 1.work_accident 2.promotion_last_5years  3.satisfaction_level  
# # # are related to retention others are not effected so much

# grouped = df.groupby(["salary","left"]).size().unstack(fill_value=0)
# # left       0     1
# # salary
# # high    1155    82
# # low     5144  2172
# # medium  5129  1317

# ax = grouped.plot(kind="bar")
# ax.set_xlabel("salary")
# ax.set_ylabel("left")
# mpl.show()
# # from above graph SALAry also influence the retention rate

required_df = df[["salary","promotion_last_5years","Work_accident","satisfaction_level"]]
dummies = pd.get_dummies(df["salary"]).astype(int)
required_df_with_dummies = pd.concat([required_df,dummies],axis="columns")
required_df_with_dummies = required_df_with_dummies.drop(['salary',"high"],axis="columns")

y = df["left"]
x = required_df_with_dummies
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = LogisticRegression()
model.fit(x_train,y_train)
model.predict(x_test)
print(model.score(x_test,y_test))
