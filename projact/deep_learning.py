from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import  layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import pandas as pd
def network_Classification():
    data = pd.read_csv("./data/predictive_maintenance.csv")
    data = data.drop(["UDI","Product ID"],axis=1)
    data_obj = data.select_dtypes(include="object") 
    data_num = data.select_dtypes(exclude="object")
    y = data["Failure Type"]
 
    le =OneHotEncoder(sparse_output=False)
    data_obj_df = pd.DataFrame(le.fit_transform(data_obj),columns=le.get_feature_names_out(data_obj.columns))
    ready_data = pd.concat([data_obj_df,data_num], axis=1)
    x = ready_data
    le2 = LabelEncoder()
    y_2 = le2.fit_transform(y)
    y_3 = tf.keras.utils.to_categorical(y_2)
    X_tran , X_test, Y_tran, Y_test = train_test_split(x,y_3,shuffle=True,random_state=42,test_size=0.2)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(9,activation = "sigmoid"),
        tf.keras.layers.Dense(128,activation = "sigmoid"),
        tf.keras.layers.Dense(64,activation = "sigmoid"),
        tf.keras.layers.Dense(32,activation = "sigmoid"),
        tf.keras.layers.Dense(8,activation = "sigmoid"),
        tf.keras.layers.Dense(64,activation = "sigmoid"),
        tf.keras.layers.Dense(8,activation = "sigmoid"),
        tf.keras.layers.Dense(6,activation = "softmax")
    ])
    model.compile(optimizer ="adam",loss = tf.keras.losses.CategoricalCrossentropy(), metrics = ["accuracy"])
    print()
    model.fit(X_tran,Y_tran,epochs= 20,validation_split= 0.2,batch_size = 32)

    model.evaluate(X_test,Y_test)

def network_Regressor():
    data = pd.read_csv("./data/Student_Performance.csv")
    data = data.drop_duplicates()
    le = LabelEncoder()
    data["Extracurricular Activities"] = le.fit_transform(data["Extracurricular Activities"])
    x = data.drop(["Performance Index"],axis =1)
    y = data["Performance Index"]
    X_tran , X_test, Y_tran, Y_test = train_test_split(x,y,shuffle=True,random_state=42,test_size=0.2)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(6,activation = "relu"),
        tf.keras.layers.Dense(256,activation = "relu"),
        tf.keras.layers.Dense(128,activation = "relu"),
        tf.keras.layers.Dense(64,activation = "relu"),
        tf.keras.layers.Dense(32,activation = "relu"),
        tf.keras.layers.Dense(8,activation = "relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer ="adam",loss = tf.keras.losses.MeanAbsoluteError(),metrics = ["mae", "mse"])
    model.fit(X_tran, Y_tran,epochs =20,validation_split=0.2, batch_size = 32)
    print()
    model.evaluate(X_test, Y_test)
    
def CNN():
    data = keras.datasets.mnist.load_data(path="mnist.npz")
    (X_train, Y_train)  , (X_test, Y_test)  = data
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    model = models.Sequential([
           layers.Input(shape=(28,28,1)),
           layers.Conv2D(32 , (3,3) , padding='same'),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(64, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(128, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Flatten(),
           layers.Dense(256, activation = "relu"),
           layers.Dense(128, activation = "relu"),
           layers.Dense(64, activation = "relu"),
            layers.Dense(10, activation = "softmax"),

    ])
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy" , metrics = ["accuracy"])
    model.fit(X_train,Y_train ,epochs = 50)
    model.save("CNN_2.keras")
def CNN_3():
    folder_train2 = ImageDataGenerator(rescale = 1./255) 
    folder_test2 = ImageDataGenerator(rescale = 1./255) 
    ready_train = folder_train2.flow_from_directory(
    "./data/data1/train",          
    target_size=(128,128),  
    class_mode="binary"     
    )
    ready_test = folder_test2.flow_from_directory(
    "./data/data1/test",          
    target_size=(128,128),  
    class_mode="binary"     
    )
    model = models.Sequential([
           layers.Input(shape=(128,128,3)),
           layers.Conv2D(32 , (3,3) , padding='same'),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(64, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(128, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Flatten(),
           layers.Dense(256, activation = "relu"),
           layers.Dense(128, activation = "relu"),
           layers.Dense(64, activation = "relu"),
            layers.Dense(2, activation = "softmax"),

    ])
    model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy" , metrics = ["accuracy"])
    model.fit(ready_train ,epochs = 30,)
    model.save("CNN_3.keras")

CNN_3()