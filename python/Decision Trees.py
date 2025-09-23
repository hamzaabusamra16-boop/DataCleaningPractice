import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

import seaborn as sns

def Clean_Data(data):
    obj_data = data.select_dtypes(include = "object")
    num_data = data.select_dtypes(exclude = "object")
    le = LabelEncoder()
    for i in obj_data:
        obj_data[i] = le.fit_transform(obj_data[i])
    ready_data = pd.concat([obj_data,num_data],axis=1)
    return ready_data
def tree(data):
    model = DecisionTreeClassifier(min_samples_leaf=10)
    x = data.drop(["Drug"],axis=1)
    y = data["Drug"]
    X_tran ,X_test ,Y_tran ,Y_test = train_test_split(x,y,test_size=0.02,random_state=42,shuffle=True)
    model.fit(X_tran,Y_tran)
    print("tree: ",model.score(X_test,Y_test))
def randomforest(data):
    model = RandomForestClassifier(min_samples_leaf= 10)
    x = data.drop(["Drug"],axis=1)
    y = data["Drug"]
    X_tran ,X_test ,Y_tran ,Y_test = train_test_split(x,y,test_size=0.02,random_state=42,shuffle=True)
    model.fit(X_tran,Y_tran)
    print("randomforest: ",model.score(X_test,Y_test))

def KNN(data):
    model = KNeighborsClassifier(n_neighbors=51)
    x = data.drop(["Drug"],axis=1)
    y = data["Drug"]
    X_tran ,X_test ,Y_tran ,Y_test = train_test_split(x,y,test_size=0.02,random_state=42,shuffle=True)
    model.fit(X_tran,Y_tran)
    print("KNN: ",model.score(X_test,Y_test))
def Regressor(data):
    model = DecisionTreeRegressor(min_samples_leaf = 10)
    x = data.drop(["Drug"],axis=1)
    y = data["Drug"]
    X_tran ,X_test ,Y_tran ,Y_test = train_test_split(x,y,test_size=0.02,random_state=42,shuffle=True)
    model.fit(X_tran,Y_tran)
    print("Regressor: ",model.score(X_test,Y_test))
def main():
    data = pd.read_csv("./data/drug200.csv")
    return data



done_data = main()
ready = Clean_Data(done_data)
Regressor(ready)
KNN(ready)
randomforest(ready)
tree(ready)
