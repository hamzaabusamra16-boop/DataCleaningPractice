from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np



def calssification():
    data = pd.read_csv("./data/insurance_classification_10000.csv")
    obj_data = data.select_dtypes(include="object")
    unm_data = data.select_dtypes(exclude="object")
    mapping = {"low": 0, "medium": 1, "high": 2}
    obj_data["charges_category"] = obj_data["charges_category"].map(mapping)
    le = LabelEncoder()
    for i in obj_data :
        if i != "charges_category":
            obj_data[i] = le.fit_transform(obj_data[i])
    ready_data = pd.concat([unm_data,obj_data],axis=1)
    x = ready_data.drop(["charges_category"],axis=1)
    y = ready_data["charges_category"]
    X_train ,X_test , Y_train , Y_test = train_test_split(x,y ,test_size=0.2,random_state=42,shuffle=True)

    nn = MLPClassifier(max_iter=10000 ,  shuffle=True , hidden_layer_sizes=(8,128,64,32))
    nn.fit(X_train,Y_train)

    print(nn.score(X_test,Y_test))
    print(nn.score(x,y))

def Regressor():
    data = pd.read_csv("./data/insurance_10000.csv")
    obj_data = data.select_dtypes(include="object")
    unm_data = data.select_dtypes(exclude="object")
    le = LabelEncoder()
    for i in obj_data :
        obj_data[i] = le.fit_transform(obj_data[i])
    ready_data = pd.concat([unm_data,obj_data],axis=1)
    x = ready_data.drop(["charges"],axis=1)
    y = ready_data["charges"]
    X_train ,X_test , Y_train , Y_test = train_test_split(x,y ,test_size=0.2,random_state=42,shuffle=True)

    nn = MLPRegressor(max_iter=10000 , shuffle=True , hidden_layer_sizes=(8,128,64,32,8))
    nn.fit(X_train,Y_train)

    print(nn.score(X_test,Y_test))
    print(nn.score(x,y))

def Network_Classification():
    data = pd.read_csv("./data/insurance_classification_10000.csv")

    # فصل الأعمدة النصية غير الهدف
    obj_data = data.select_dtypes(include="object").drop(columns=["charges_category"])
    unm_data = data.select_dtypes(exclude="object")

    # تحويل العمود الهدف إلى أرقام
    mapping = {"low": 0, "medium": 1, "high": 2}
    y = data["charges_category"].map(mapping)

    # One-Hot للأعمدة النصية الأخرى
    le = OneHotEncoder(sparse_output=False)
    obj_data_df = pd.DataFrame(le.fit_transform(obj_data), columns=le.get_feature_names_out(obj_data.columns))

    # دمج البيانات الجاهزة
    ready_data = pd.concat([unm_data, obj_data_df], axis=1)
    x = ready_data.values
    y = tf.keras.utils.to_categorical(y, num_classes=3)

    # تقسيم البيانات
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

    # بناء النموذج
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax"),
    ]) 
    model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())

    # التدريب
    model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))

def Network_Regressor():
    data = pd.read_csv("./data/insurance_10000.csv")
    obj_data = data.select_dtypes(include="object")
    unm_data = data.select_dtypes(exclude="object")
    le = OneHotEncoder(sparse_output=False,drop=None)
    obj_data_df = pd.DataFrame(le.fit_transform(obj_data),columns=le.get_feature_names_out(obj_data.columns))
    ready_data = pd.concat([unm_data,obj_data_df],axis=1)
    x = ready_data.drop(["charges"], axis= 1)
    y = ready_data["charges"]
    model = tf.keras.models.Sequential([
           tf.keras.layers.Dense(7,activation = "relu"),
           tf.keras.layers.Dense(128,activation = "relu"),
           tf.keras.layers.Dense(64,activation = "relu"),
           tf.keras.layers.Dense(32,activation = "relu"),
           tf.keras.layers.Dense(32,activation = "relu"),
           tf.keras.layers.Dense(1)
    ])
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)
    model.compile(optimizer = "adam",loss = tf.keras.losses.MeanSquaredError(),metrics = ["mae"])
    model.fit(X_train,Y_train,epochs =10)
    model.evaluate(X_test,Y_test)

def CNN():
    data = keras.datasets.fashion_mnist.load_data()
    (X_train , Y_train),(X_test ,Y_test) = data
    X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

    model = models.Sequential([
           layers.Conv2D(32, (3,3) , input_shape =(28,28,1)),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(64, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Conv2D(128, (3,3)),
           layers.MaxPooling2D(2,2),
           layers.Flatten(),
           layers.Dense(256, activation = "relu"),
           layers.Dense(128, activation = "relu"),
           layers.Dense(64, activation = "relu"),
           layers.Dense(32, activation = "relu"),
           layers.Dense(10, activation = "softmax")
    ])
    model.compile(optimizer ="adam",loss = "sparse_categorical_crossentropy",metrics = ["accuracy"])
    model.fit(X_train,Y_train,epochs = 40)
    model.save("CNN.keras")
def load_CNN():
    model_CNN = tf.keras.models.load_model("CNN.keras")
    (_, _), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
    test = model_CNN.predict (X_test[7:8])
    test_class = np.argmax(test)
    X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
    test_loss, test_acc = model_CNN.evaluate(X_test, Y_test)
    print(f"Test accuracy: {test_acc}")
    print(f"Predicted class for first image: {test_class}, True class: {Y_test[7:8]}")
load_CNN()