import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

def projact_one():
     
   data = pd.read_csv("./data/housing.csv")

   model1 = LinearRegression()
   model2 = KNeighborsRegressor()
   model3 = RandomForestRegressor()
   model4 = DecisionTreeRegressor()
   model = VotingRegressor([("Linear",model1),("KN",model2),("RandomForest",model3),("tree",model4)])
   x = data.drop(["MEDV"],axis=1)
   y = data["MEDV"]
   data = data.sort_values(by =["RM"])
   X_train ,X_test , Y_train , Y_test = train_test_split(x,y ,test_size=0.2,random_state=42,shuffle=True)
   model.fit(X_train,Y_train)
   print(model.score(X_test,Y_test))
   data = data.sort_values(by="RM")
   plt.scatter(X_test["RM"],Y_test,marker="o",color = "r")
   plt.grid()
   plt.xlabel("RM")
   plt.ylabel("MEDV")
   plt.title("RM VS MEDV")
   plt.show()
def projact_two():
    data = pd.read_csv("./data/iris.csv")
    le = LabelEncoder()
    data["species"] = le.fit_transform(data["species"])
    model1 = LogisticRegression()
    model2 = KNeighborsClassifier()
    model3 = RandomForestClassifier()
    model4 = DecisionTreeClassifier()
    model = VotingClassifier([("classification",model1),("KN",model2),("RandomForest",model3),("tree",model4)])

    x = data.drop(["species"],axis=1)
    y = data["species"]
 
    X_train ,X_test , Y_train , Y_test = train_test_split(x,y ,test_size=0.2,random_state=42,shuffle=True)
    model.fit(X_train,Y_train)
    print(model.score(X_test,Y_test))
    plt.scatter(X_test["petal_length"],Y_test,marker="o",color = "r")
    plt.grid()
    plt.xlabel("petal_length")
    plt.ylabel("species")
    plt.title("petal_length VS species")
    plt.show()

projact_one()
projact_two()
