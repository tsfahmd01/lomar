from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import os
import socket
import pickle
from sklearn.model_selection import train_test_split
import zlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D, BatchNormalization
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt 

main = tkinter.Tk()
main.title("LoMar: A Local Defense Against Poisoning Attack on Federated Learning") #designing main screen
main.geometry("1300x1200")

global filename, propose_size, extension_size, acc, lomar_acc, X, Y, X_train, X_test, y_train, y_test


def upload():
    text.delete('1.0', END)
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    dataset = pd.read_csv("Dataset/mnist.csv")
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))

def processDataset():
    global X, Y, X_train, X_test, y_train, y_test
    global dataset
    text.delete('1.0', END)
    dataset = dataset.values
    Y = dataset[:,0]
    labels = np.unique(Y)
    X = dataset[:,1:dataset.shape[1]]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = to_categorical(Y)
    X = np.reshape(X, (X.shape[0], 28, 28, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test
    text.insert(END,"Dataset shuffling & normalization processing completed\n\n")
    text.insert(END,"Total records found in Dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Total labels found in Dataset : "+str(labels.tolist())+"\n\n")
    text.insert(END,"Dataset Training & Testing Size Details\n")
    text.insert(END,"80% dataset for training : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset for testing : "+str(X_test.shape[1])+"\n")

def uploadGenuine():
    global X, Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    global propose_size, extension_size, acc, lomar_acc
    cnn_model = Sequential()
    cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units = 256, activation = 'relu'))
    cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    if os.path.exists("model/cnn_weights.keras") == False:
        model_check_point = ModelCheckpoint(filepath='model/cnn_weights.keras', verbose = 1, save_best_only = True)
        hist = cnn_model.fit(X_train, y_train, batch_size = 16, epochs = 5, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/cnn_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        cnn_model = load_model("model/cnn_weights.keras")
    predict = cnn_model.predict(X_test)
    y_test1 = np.argmax(y_test, axis=1)
    predict = np.argmax(predict, axis=1)  
    lomar_acc = accuracy_score(predict, y_test1)
    with open("model/cnn_weights.keras", "rb") as file:
        model = file.read()
    file.close()
    propose_size = len(model)
    compressed_data = zlib.compress(model)
    extension_size = len(compressed_data)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    s.sendall(len(compressed_data).to_bytes(4, byteorder='big') + compressed_data)
    data = s.recv(100)
    data = data.decode()
    text.insert(END,"Server Response : "+data+"\n")
    text.insert(END,"LoMar Defense Accuracy : "+str(lomar_acc)+"\n\n")
    s.close()

def uploadPoison():
    global X_train, X_test, y_train, y_test
    global propose_size, extension_size, acc, lomar_acc
    y_train = np.argmax(y_train, axis=1)
    poison = []
    count = 0
    for i in range(len(y_train)):
        if y_train[i] == 1 and count < 4000:#here we are checking if training label is 1
            poison.append(0) #then we are poisoning 1 label with 0
            count += 1
        else:
            poison.append(y_train[i])    
    poison = np.asarray(poison)
    y_train = to_categorical(poison)
    poison_model = Sequential()
    poison_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    poison_model.add(MaxPooling2D(pool_size = (2, 2)))
    poison_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
    poison_model.add(MaxPooling2D(pool_size = (2, 2)))
    poison_model.add(Flatten())
    poison_model.add(Dense(units = 256, activation = 'relu'))
    poison_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    poison_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])  
    if os.path.exists("model/poison_weights.keras") == False:
        model_check_point = ModelCheckpoint(filepath='model/poison_weights.keras', verbose = 1, save_best_only = True)
        hist = poison_model.fit(X_train, y_train, batch_size = 16, epochs = 5, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/poison_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        poison_model = load_model("model/poison_weights.keras")
    predict = poison_model.predict(X_test)
    y_test1 = np.argmax(y_test, axis=1)
    predict = np.argmax(predict, axis=1)  
    acc = accuracy_score(predict, y_test1)
    with open("model/poison_weights.keras", "rb") as file:
        model = file.read()
    file.close()
    propose_size = len(model)
    compressed_data = zlib.compress(model)
    extension_size = len(compressed_data)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    s.sendall(len(compressed_data).to_bytes(4, byteorder='big') + compressed_data)
    data = s.recv(100)
    data = data.decode()
    text.insert(END,"Server Response : "+data+"\n")
    text.insert(END,"No Defence Accuracy : "+str(acc)+"\n\n")
    s.close()
    
def lomarAccuracy():
    global acc, lomar_acc
    labels = ['No Defense', 'LoMar (Original Setup on MNIST)']
    height = (acc, lomar_acc)
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Setup")
    plt.ylabel("Accuracy %")
    plt.title("No Defense vs LoMar (Original Setup on MNIST) Accuracy")
    plt.xticks()
    plt.tight_layout()
    plt.show()

def extensionGraph():
    global propose_size, extension_size
    text.insert(END,"Original Model Size : "+str(propose_size)+"\n")
    text.insert(END,"Compressed Model Size (zlib Extension) : "+str(extension_size)+"\n\n")
    labels = ['Original Model', 'Compressed (zlib)']
    height = (propose_size, extension_size)
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Transmission Method")
    plt.ylabel("Model Size in Bytes")
    plt.title("Model Size Comparison: Original vs Compressed")
    plt.xticks()
    plt.tight_layout()
    plt.show()    

font = ('times', 16, 'bold')
title = Label(main, text='LoMar: A Local Defense Against Poisoning Attack on Federated Learning')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=26,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=480,y=100)
text.config(font=font1)


font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload MNIST Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=50,y=150)
processButton.config(font=font1) 

genuineButton = Button(main, text="Upload Genuine Model to Server", command=uploadGenuine)
genuineButton.place(x=50,y=200)
genuineButton.config(font=font1) 

poisonButton = Button(main, text="Upload Poison Model to Server", command=uploadPoison)
poisonButton.place(x=50,y=300)
poisonButton.config(font=font1)

accButton = Button(main, text="LoMar & No Defense Accuracy", command=lomarAccuracy)
accButton.place(x=50,y=350)
accButton.config(font=font1)

extensionButton = Button(main, text="Model Size Compression Graph", command=extensionGraph)
extensionButton.place(x=50,y=400)
extensionButton.config(font=font1)


main.config(bg='OliveDrab2')
main.mainloop()
