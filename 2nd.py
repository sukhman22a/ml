"""gradient descent 
https://youtu.be/sDv4f4s2SB8?feature=shared
https://youtu.be/vsWrXfO3wWw?feature=shared    15:21sec"""
import numpy as np
import pandas as pd
# from sklearn import linear_model
def gradient_descent(x,y):
    m_curr = b_curr = 0
    n= len(x)
    iterations = 1000
    learning_rate = 0.0002
    for i in range(iterations):
        # 
        y_predicated = m_curr*x + b_curr
        cost = (1/n)*sum([val **2 for val in (y-y_predicated)])
        md = -(2/n)*sum((y - y_predicated)*x)
        bd = -(2/n)*sum(y - y_predicated)
        m_curr -= md*learning_rate
        b_curr -= bd*learning_rate
        print("m {}, b {} cost {} iteration {}".format(m_curr,b_curr,cost,i))
    return b_curr, m_curr     

df = pd.read_csv("data/marks.csv")
# print(df)
# x = np.array([1,2,3,4,5])
# y = np.array([5,7,9,11,13])

bg ,mg = gradient_descent(df["math"],df["cs"])
# reg = linear_model.LinearRegression()
# reg.fit(df[["math"]].values,df[["cs"]])
print("bg {}    mg {}\n".format(bg,mg))
# print("b {}  m {}".format(reg.intercept_,reg.coef_))  #b [1.91521931]  m [[1.01773624]]