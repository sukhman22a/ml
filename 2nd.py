""" Saving a model """
import pandas as pd
import matplotlib.pyplot as mpl
import word2number.w2n as w2n
from sklearn import linear_model
import joblib
# import numpy as np
df = pd.read_csv("data/hiring.csv")
reg = linear_model.LinearRegression() 


# The 'experience' column may contain missing values (NaN). 
# We first replace NaN values with the string "zero" using fillna(). 
# Then, we apply the word_to_num function (from the w2n library) to convert 
# the words (e.g., "one", "two", "three") into corresponding numerical values. 
# If 'experience' is NaN, it will be treated as "zero" and converted to 0.
df["experience_in_number"] = df["experience"].fillna("zero").apply(w2n.word_to_num)
df["test_score(out of 10)"] = df["test_score(out of 10)"].fillna(df["test_score(out of 10)"].median())


reg.fit(df[["test_score(out of 10)","interview_score(out of 10)","experience_in_number"]].values,df[["salary($)"]].values)

# print(reg.predict([[9,6,2]]))
# "test_score(out of 10)","interview_score(out of 10)","experience_in_number"

joblib.dump(reg,"model/hiring_model(ml)")
# to save the model


model_hiring  = joblib.load("model/hiring_model(ml)")
# to load the model
print(model_hiring.predict([[9,6,2]]))