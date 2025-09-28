
import pandas as pd
from sklearn.preprocessing  import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import joblib

def clear_data():
    data_clear = pd.read_csv("./dataset/realtor-data.csv")
    data_clear = data_clear.drop(["brokered_by","street","prev_sold_date"],axis=1)
    num_data = data_clear.select_dtypes(exclude="object")
    obj_data = data_clear.select_dtypes(include="object")
    le = LabelEncoder()
    for i in obj_data:
        obj_data[i] = le.fit_transform(obj_data[i])
    obj_data = pd.get_dummies(obj_data, columns=obj_data.columns)
    num_data = num_data.fillna(num_data.mean())
    data = pd.concat([num_data,obj_data],axis=1)
    return data
def train_model():
    data = clear_data()
    x = data.drop(["price"],axis=1)
    y= data["price"]
    model1 = LinearRegression()
    model2 = DecisionTreeRegressor()
    model3 = RandomForestRegressor()
    model4 = KNeighborsRegressor()
    model = VotingRegressor([
           ("Linear_Reg ", model1),("Tree_Reg ", model2),("RandomFores_Reg ", model3),("KNeighbors_Reg ", model4),
    ])
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    model.fit(X_train,Y_train)
    
    joblib.dump(model , "projcat_three.pkl")
    print(model.score(X_train,Y_train))
    print(model.score(X_test,Y_test))

train_model()
    