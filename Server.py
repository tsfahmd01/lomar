import socket 
from threading import Thread 
from socketserver import ThreadingMixIn
import pickle
import numpy as np
import os
from keras.models import load_model, Model
import pickle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KernelDensity
import zlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import socket
import pickle
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Dataset/mnist.csv")
dataset.fillna(0, inplace = True)
dataset = dataset.values

Y = dataset[:,0]
X = dataset[:,1:dataset.shape[1]]

sc = StandardScaler()
X = sc.fit_transform(X)

#Y = to_categorical(Y)

X = np.reshape(X, (X.shape[0], 28, 28, 1))
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #split dataset into train and test

update_model = {}
running = True


def startCentralizedServer():
    class UpdateModel(Thread):

        
        def getScore(self, kde, data):
            scores = kde.score_samples(data)
            norm = np.linalg.norm(-scores)
            return np.mean(-scores/norm)

        def estimatedensity(self, model, X_test):
            models = Model(model.inputs, model.layers[-2].output)#create clean  model
            models = models.predict(X_test)  #extracting cnn features
            kde = KernelDensity()
            kde.fit(models[0:500,0:50])
            score = self.getScore(kde,models[0:500,0:50])
            return score


        def getAccuracy(self, model, X_test, y_test):
            predict = model.predict(X_test)
            predict = np.argmax(predict, axis=1)
            acc = accuracy_score(y_test, predict)
            return acc
 
        def __init__(self,ip,port): 
            Thread.__init__(self) 
            self.ip = ip 
            self.port = port 
            print('Request received from Client IP : '+ip+' with port no : '+str(port)+"\n") 
 
        def run(self): 
            header = conn.recv(4)
            if not header:
                return
            data_len = int.from_bytes(header, byteorder='big')
            chunks = []
            bytes_recd = 0
            while bytes_recd < data_len:
                chunk = conn.recv(min(data_len - bytes_recd, 65536))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_recd += len(chunk)
            data = b''.join(chunks)
            model = zlib.decompress(data)
            status = "Pending"
            if os.path.exists("globalModel/model.keras"):
                existing = load_model("globalModel/model.keras")
                with open("test.keras", "wb") as file:
                    file.write(model)
                file.close()
                received = load_model("test.keras")
                ex_acc = self.getAccuracy(existing, X_test, y_test)
                rec_acc = self.getAccuracy(received, X_test, y_test)
                kde1 = self.estimatedensity(existing, X_test)
                kde2 = self.estimatedensity(received, X_test)
                if ex_acc > 0.95 and rec_acc > 0.95 or kde1 == kde2:
                    with open("globalModel/model.keras", "wb") as file:
                        file.write(model)
                    file.close()
                    status = "Genuine Model Received and Updated to server"
                    print("Accuracy with LoMar : "+str(ex_acc))
                else:
                    print("Accuracy with No Defence : "+str(rec_acc))
                    status = "Poison Model Received and Ignored Updation"
            else:
                with open("globalModel/model.keras", "wb") as file:
                    file.write(model)
                file.close()
                existing = load_model("globalModel/model.keras")
                ex_acc = self.getAccuracy(existing, X_test, y_test)
                print("Accuracy with LoMar : "+str(ex_acc))
                status = "Genuine Model Received and Updated to server"            
            conn.send(status.encode())         
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
    server.bind(('localhost', 2222))
    print("Centralized Server Started & waiting for incoming connections\n\n")
    while running:
        server.listen(4)
        (conn, (ip,port)) = server.accept()
        newthread = UpdateModel(ip,port) 
        newthread.start() 
    
def startServer():
    Thread(target=startCentralizedServer).start()

startServer()

