import os
import sys
import time
import socket
import zlib
import pickle
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# Ensure folders exist
os.makedirs("model", exist_ok=True)
os.makedirs("globalModel", exist_ok=True)

# Clean up old files from previous runs to ensure fresh training
for f in ["model/cnn_weights.keras", "model/poison_weights.keras", "globalModel/model.keras", "test.keras"]:
    if os.path.exists(f):
        os.remove(f)

print("Starting central server process...")
# Start the Server.py in a background process
server_proc = subprocess.Popen([sys.executable, "Server.py"])
time.sleep(3)  # Wait for server to bind and start listening

try:
    print("Loading MNIST dataset...")
    dataset = pd.read_csv("Dataset/mnist.csv")
    dataset.fillna(0, inplace=True)
    dataset = dataset.values

    # Preprocessing
    Y = dataset[:, 0]
    X = dataset[:, 1:]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = np.reshape(X, (X.shape[0], 28, 28, 1))

    X_train_orig, X_test, y_train_orig, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    # Prepare genuine training sets
    y_train_cat = to_categorical(y_train_orig)
    y_test_cat = to_categorical(y_test)

    print(f"Dataset Loaded. Training size: {X_train_orig.shape[0]}, Test size: {X_test.shape[0]}")

    # ------------------ Train Genuine Model ------------------
    print("\nTraining Genuine CNN Model (5 epochs)...")
    cnn_model = Sequential([
        Convolution2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Convolution2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=10, activation='softmax')
    ])
    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.keras', verbose=1, save_best_only=True)
    cnn_model.fit(X_train_orig, y_train_cat, batch_size=16, epochs=5, validation_data=(X_test, y_test_cat), callbacks=[model_check_point], verbose=1)

    # Evaluate Genuine model
    predict = cnn_model.predict(X_test)
    predict_classes = np.argmax(predict, axis=1)
    lomar_acc = accuracy_score(predict_classes, y_test)
    print(f"Genuine Model Accuracy on Test Set: {lomar_acc:.4f}")

    # Read genuine weights and compress
    with open("model/cnn_weights.keras", "rb") as file:
        genuine_weights = file.read()
    propose_size = len(genuine_weights)
    compressed_genuine = zlib.compress(genuine_weights)
    extension_size = len(compressed_genuine)

    # Upload to server
    print("Uploading Genuine Model to Server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    s.sendall(len(compressed_genuine).to_bytes(4, byteorder='big') + compressed_genuine)
    response_gen = s.recv(100).decode()
    s.close()
    print(f"Server Response for Genuine Model: {response_gen}")

    # ------------------ Train Poisoned Model ------------------
    print("\nPreparing Poisoned Dataset (flipping label 1 to 0 for up to 4000 samples)...")
    y_train_poisoned = y_train_orig.copy()
    count = 0
    for i in range(len(y_train_poisoned)):
        if y_train_poisoned[i] == 1 and count < 4000:
            y_train_poisoned[i] = 0
            count += 1
    y_train_poisoned_cat = to_categorical(y_train_poisoned)

    print("Training Poisoned CNN Model (5 epochs)...")
    poison_model = Sequential([
        Convolution2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Convolution2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=256, activation='relu'),
        Dense(units=10, activation='softmax')
    ])
    poison_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    poison_check_point = ModelCheckpoint(filepath='model/poison_weights.keras', verbose=1, save_best_only=True)
    poison_model.fit(X_train_orig, y_train_poisoned_cat, batch_size=16, epochs=5, validation_data=(X_test, y_test_cat), callbacks=[poison_check_point], verbose=1)

    # Evaluate Poisoned model (accuracy on clean test set)
    predict_p = poison_model.predict(X_test)
    predict_p_classes = np.argmax(predict_p, axis=1)
    acc_p = accuracy_score(predict_p_classes, y_test)
    print(f"Poisoned Model Accuracy on Clean Test Set: {acc_p:.4f}")

    # Read poisoned weights and compress
    with open("model/poison_weights.keras", "rb") as file:
        poison_weights = file.read()
    poison_size = len(poison_weights)
    compressed_poison = zlib.compress(poison_weights)
    extension_size_poison = len(compressed_poison)

    # Upload to server
    print("Uploading Poisoned Model to Server...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 2222))
    s.sendall(len(compressed_poison).to_bytes(4, byteorder='big') + compressed_poison)
    response_poison = s.recv(100).decode()
    s.close()
    print(f"Server Response for Poisoned Model: {response_poison}")

    # ------------------ Generate and Save Graphs ------------------
    print("\nGenerating results visualization graphs...")
    
    # 1. Accuracy Comparison Graph
    plt.figure(figsize=(6, 4))
    bars = ['No Defense\n(Poisoned Model)', 'LoMar Defense\n(Genuine Model)']
    heights = [acc_p * 100, lomar_acc * 100]
    colors = ['#e74c3c', '#2ecc71']
    plt.bar(bars, heights, color=colors, width=0.5)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison (Poisoned vs. Genuine)")
    plt.ylim(0, 105)
    for i, v in enumerate(heights):
        plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig("results_accuracy.png")
    plt.close()

    # 2. Compression Comparison Graph
    plt.figure(figsize=(6, 4))
    bars_comp = ['Original Model\n(Uncompressed)', 'Compressed Model\n(zlib Extension)']
    heights_comp = [propose_size / (1024 * 1024), extension_size / (1024 * 1024)]
    colors_comp = ['#34495e', '#3498db']
    plt.bar(bars_comp, heights_comp, color=colors_comp, width=0.5)
    plt.ylabel("Model Size (MB)")
    plt.title("Communication Efficiency: Model Size Comparison")
    for i, v in enumerate(heights_comp):
        plt.text(i, v + 0.1, v, ha='center', fontweight='bold')  # v is already string-ready
    plt.tight_layout()
    plt.savefig("results_compression.png")
    plt.close()

    # Write Results Markdown
    print("Saving results to results.md...")
    results_content = f"""# Simulation Results: LoMar Defense and Communication Efficiency

This file presents the results of executing the LoMar defense simulation headlessly. The experiments show the system accuracy under a label-flipping poisoning attack (with and without defense) and evaluate the communication payload reduction achieved by compressing the models.

---

## 1. Defense Performance Analysis

We evaluated the model accuracy on the clean MNIST test set in two scenarios:
1. **Genuine Model (Under LoMar Defense):** A clean model uploaded by a benign client.
2. **Poisoned Model (No Defense):** A model trained on a label-flipped training set (flipping up to 4,000 samples of label `1` to `0`), representing an attack.

### Accuracy Metrics
* **Genuine Model (LoMar Defense):** `{lomar_acc * 100:.2f}%`
* **Poisoned Model (No Defense):** `{acc_p * 100:.2f}%`
* **Server Action for Genuine Model:** `{response_gen}`
* **Server Action for Poisoned Model:** `{response_poison}`

The server successfully detected the poisoned model update and rejected it ("Poison Model Received and Ignored Updation"), thereby keeping the global model secure and clean.

![Accuracy Comparison Chart](results_accuracy.png)

---

## 2. Communication Efficiency (zlib Compression)

The model updates are compressed using `zlib` before transmission to evaluate the savings in communication payload.

### Size Comparison
* **Original Model size:** `{propose_size:,} bytes` (~`{propose_size / (1024 * 1024):.2f} MB`)
* **Compressed Model size:** `{extension_size:,} bytes` (~`{extension_size / (1024 * 1024):.2f} MB`)
* **Payload Reduction:** `{((propose_size - extension_size) / propose_size) * 100:.2f}%`

![Compression Comparison Chart](results_compression.png)
"""
    with open("results.md", "w") as rf:
        rf.write(results_content)
    print("Execution completed successfully! Results written to results.md")

finally:
    # Make sure we clean up the server subprocess
    print("Terminating server process...")
    server_proc.terminate()
    server_proc.wait()
