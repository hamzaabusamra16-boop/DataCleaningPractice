import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

def classification(data):
    data_one = pd.read_csv(data)
    data_one = data_one.drop(["Unnamed: 32","id"],axis=1)
    le = LabelEncoder()
    data_one["diagnosis"] = le.fit_transform(data_one["diagnosis"])
    x= data_one.drop(["diagnosis"],axis=1)
    y= data_one["diagnosis"]
    X_tran , X_test , Y_tran , Y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
    model = LogisticRegression(max_iter=500,tol=0.01)
    model.fit(X_tran,Y_tran)
    print(model.score(X_test,Y_test))
def lLinear(data):
    data_two = pd.read_csv(data)
    data_two = data_two.drop_duplicates()
    obj_data = data_two.select_dtypes(include="object")
    num_data = data_two.select_dtypes(exclude="object")
    le = LabelEncoder()
    for i in obj_data:
        obj_data[i] = le.fit_transform(obj_data[i])
    ready_data = pd.concat([obj_data,num_data],axis=1)
    model = LinearRegression()
    x= ready_data.drop(["charges"],axis=1)
    y= ready_data["charges"]
    X_tran , X_test , Y_tran , Y_test = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
    model.fit(X_tran,Y_tran)
    print(model.score(X_test,Y_test))
classification("./data/data.csv")
lLinear("./data/insurance.csv")