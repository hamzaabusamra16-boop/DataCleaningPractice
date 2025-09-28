import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import  layers, models
from sklearn.preprocessing import LabelEncoder

def clrean_data():
    data = ImageDataGenerator(rescale =1./255, validation_split=0.2,rotation_range=40,width_shift_range=0.2,
    height_shift_range=0.2,horizontal_flip=True, fill_mode='nearest',zoom_range=0.2, shear_range=0.2
)
    train_data = data.flow_from_directory(
        "./dataset/archive-5/animals/animals",
        target_size=(128,128),
        batch_size=1,
        class_mode="categorical",
        subset="training"
    )
    val_data = data.flow_from_directory(
        "./dataset/archive-5/animals/animals",
        target_size = (128,128),
        batch_size = 1,
        class_mode = "categorical",
        subset = "validation"
    )
    return train_data,val_data
def train_model():
        train_data,val_data = clrean_data()
        model = models.Sequential([
        layers.Input(shape = (128,128, 3)),
        layers.Conv2D(64,(3,3),padding = 'same'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128,(3,3),padding = 'same'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(256,(3,3),padding = 'same'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(90, activation="softmax"),
        ]
        )
        ## نموذج رح يفشل بسبب قلة عدد الصور وكثر عدد فئات 
        model.compile(optimizer = "adam",loss="categorical_crossentropy",metrics = ["accuracy"])
        model.fit(train_data,validation_data=val_data,epochs=100)
        model.save("Animals.kerase")


train_model()