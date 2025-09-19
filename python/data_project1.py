import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 

data = pd.read_csv("./data/data1.csv")
data_num = data.select_dtypes(exclude="object")
data["adult_male_str"] = data["adult_male"].astype(str)
data["alone_str"] = data["alone"].astype(str)

data_object = data.select_dtypes(include="object")
data_num = data_num.fillna(data_num.mean())
for i in data_object:
    data_object[i] = data_object[i].fillna(data_object[i].mode()[0])
le = LabelEncoder()
for i in data_object:
    data_object[i] = le.fit_transform(data_object[i])
ready_data = pd.concat([data_num,data_object],axis=1)
ready_data = ready_data.drop(['adult_male','alone'],axis=1)
reashaping =ready_data.corr()
#sns.heatmap(reashaping)
print(ready_data.head())
x = ready_data["survived"]
y= ready_data["alive"]
ready_data = ready_data.sort_values(by=["survived", "age"])
ready_data = ready_data.drop_duplicates()
plt.scatter(x,y)
plt.xlabel("Survived")
plt.ylabel("Alive")
plt.title("Survived VS Alive")

plt.grid()

plt.show()