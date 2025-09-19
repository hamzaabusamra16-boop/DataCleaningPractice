import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 

data = pd.read_csv("./data/data3.csv")
data = data.drop_duplicates()
le=LabelEncoder()
data_partone = data.select_dtypes(exclude="object")
data_partTow = data.select_dtypes(include="object")

data_partone = data_partone.fillna(data_partone.mean())
for i in data_partTow:
    data_partTow[i]= le.fit_transform(data_partTow[i])
    data_partTow[i]= data_partTow[i].fillna(data_partTow[i].mode()[0])
ready_data = pd.concat([data_partone,data_partTow],axis= 1)
cor = ready_data.corr()
ready_data = ready_data.sort_values(by= ["IMDB_title_ID","Release_year"])
print(ready_data)
x = ready_data["IMDB_title_ID"]
y = ready_data["Release_year"]
plt.xlabel("IMDB_title_ID")
plt.ylabel("Release_year")
plt.title("IMDB_title_ID VS Release_year")
plt.plot(x,y,color="y",linewidth=1)
plt.grid()
plt.show()
