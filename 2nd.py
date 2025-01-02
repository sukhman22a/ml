"""MULtinomial Navie Bayes
a probabilistic classifier based on Bayes’ theorem 
specifically designed for discrete data, particularly text data.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/spam.csv")
# print(df.groupby("Category").describe())

df["spam"] = df["Category"].apply(lambda x : 0 if 'ham' == x else 1)
df = df.drop(["Category"],axis="columns")
x_train ,x_test ,y_train ,y_test = train_test_split(df["Message"],df["spam"],test_size=0.3)

clf = Pipeline([
    ("verctorizer",CountVectorizer()),
    ("mnb",MultinomialNB()),
    ("score", cross_val_score())
])
clf.fit(x_train,y_train)
print(clf.score(x_test,y_test))