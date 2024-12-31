""""""
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as mpl
iris = load_iris()
# print(dir(iris))
# ['DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names']
df = pd.DataFrame(iris.data, columns=iris.feature_names)
x = df[["petal length (cm)","petal width (cm)"]]
scale = MinMaxScaler()
scale_data = pd.DataFrame(scale.fit_transform(x),columns=["pl cm", "pw cm"])

# mpl.scatter(scale_data['pl cm'],scale_data["pw cm"])
# mpl.show()

km = KMeans(n_init="auto",n_clusters=5)
y_predicted = km.fit_predict(scale_data)
scale_data['clusters'] = y_predicted
# print(km.cluster_centers_)

# df0 = scale_data[scale_data["clusters"] == 0]
# df1 = scale_data[scale_data["clusters"] == 1]
# df2 = scale_data[scale_data["clusters"] == 2]
# df3 = scale_data[scale_data["clusters"] == 3]
# df4 = scale_data[scale_data["clusters"] == 4]

# mpl.scatter(df4["pw cm"],df4["pl cm"],color = 'brown')
# mpl.scatter(df0["pw cm"],df0["pl cm"],color = "red")
# mpl.scatter(df1["pw cm"],df1["pl cm"],color = "yellow")
# mpl.scatter(df2["pw cm"],df2["pl cm"],color = "blue")
# mpl.scatter(df3["pw cm"],df3["pl cm"],color = "black")
# mpl.legend()
# mpl.show()


sse = [] #sum of square  of errors
for k in range(1,11):
    kp = KMeans(n_init=10,n_clusters=k)
    kp.fit(scale_data.drop(["clusters"],axis="columns"))
    sse.append(kp.inertia_)

mpl.plot(range(1,11),sse)
mpl.show()
