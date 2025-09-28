import pandas as pd
import glob
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing  import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def  ready_data():
    folder = "./dataset/dataset_for_projactOne/*.csv"
    All_files = glob.glob(folder)
    dfs = [pd.read_csv(files)for files in All_files] 
    data = pd.concat(dfs , ignore_index=True)
    data = data.drop_duplicates()
    tokenizer = Tokenizer()
    x = data.drop(["Category"],axis=1)
    x= x["Title"].astype(str).tolist()
    y = data["Category"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    y =to_categorical(y)
    print(le.classes_)
    tokenizer.fit_on_texts(x)
    Sequences = tokenizer.texts_to_sequences(x)
    x_sqr = pad_sequences(Sequences, maxlen = 1000)
    return x_sqr,y
def train_data():
    x_sqr,y  = ready_data()  
    model = Sequential()
    model.add(Embedding(input_dim = 203466 , output_dim =246 ))
    model.add(SimpleRNN(128))
    model.add(Dense(20 , activation= "softmax"))
    model.compile(optimizer = "adam" , loss ="categorical_crossentropy", metrics = ["accuracy"])
    X_train, X_test, Y_train, Y_test = train_test_split(x_sqr, y, test_size=0.2, random_state=42, shuffle=True)
    ## شغلت نمودج بس ما كملتو بسبب مدة وقت عالي لتدريب حلو ٢٠ ساعة 
    model.fit(X_train,Y_train , epochs = 10, validation_data = (X_test,Y_test))
    model.save("projactOne.keras")
train_data()