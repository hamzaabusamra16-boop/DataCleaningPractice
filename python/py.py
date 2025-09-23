import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import seaborn as sns
data = pd.read_csv("./data/penguins.csv")
data = data.dropna()
le = LabelEncoder()
data["sex"] = le.fit_transform(data["sex"])
def draw_Cluster():
    list = []
    for i in range(2,31):
        model= KMeans(n_clusters=i,max_iter=500,random_state=42)
        model.fit(data)
        list.append(model.inertia_)
    plt.plot(range(2,31),list,marker="X")
    plt.show()
def Cluster(data):
    model= KMeans(n_clusters=6,max_iter=500,random_state=42)
    model.fit(data)
    data["Cluster"]= model.labels_
    print(data)
    def classification(data):
        x = data.drop(["Cluster"],axis=1)
        y = data["Cluster"]
        X_tran, X_test, Y_tran ,Y_test = train_test_split(x,y ,test_size=0.02,random_state=42,shuffle=True)
        model_tow = LogisticRegression(max_iter=10000,tol=0.01)
        model_tow.fit(X_tran, Y_tran)
        print(model_tow.score(X_test, Y_test))
        plt.hist(y)
        plt.show()
    classification(data)    
Cluster(data)