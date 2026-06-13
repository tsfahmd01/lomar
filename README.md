# LoMar (Reproduction & Extension on MNIST)

This repository contains a **reproduction and evaluation** of the **LoMar (Local Malicious Factor)** defense algorithm, originally proposed by Li et al. (2023).

This project reproduces the defense mechanism using a client-server socket architecture, evaluates it on the **MNIST handwritten digit dataset** under label-flipping attacks, and extends the implementation with a **communication-efficiency layer** using `zlib` compression.

## Authors & Contributors
* **Mohammed Tauseef Ahmed**
* **Eppa Srujan Reddy**
* **Shaik Shoaib Hannan**

*Submitted to JNTUH, Hyderabad in partial fulfillment of the academic requirements for the award of the degree of Bachelor of Technology in Computer Science and Engineering (AI&ML).* 
*Presented at the 5th International Conference on Recent Trends in Engineering, Technology and Management 2025 (ICRETM'25).*

---

## Original Research Citation

If you use or build upon this codebase, please cite the original authors of the LoMar defense algorithm:

> **LoMar: A Local Defense Against Poisoning Attack on Federated Learning**
> Xingyu Li, Zhe Qu, Shangqing Zhao, Bo Tang, Zhuo Lu, and Yao Liu
> *IEEE Transactions on Dependable and Secure Computing (TDSC)*, Vol. 20, No. 4, pp. 3155-3169, 2023.
> DOI: [10.1109/TDSC.2021.3129272](https://doi.org/10.1109/TDSC.2021.3129272)

---

## System Requirements & Installation

### 1. Install System-Level GUI Package (Tkinter)
Since the client application utilizes a graphical user interface (GUI) built with Tkinter, you must install the Python Tkinter package on your system:
```bash
sudo apt-get update
sudo apt-get install -y python3-tk
```

### 2. Setup Virtual Environment
Create and activate a Python virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies
Install the required packages in your virtual environment:
```bash
pip install --upgrade pip
pip install pandas tensorflow scikit-learn matplotlib
```

---

## Code Compatibility Adjustments

Because of Keras updates in newer TensorFlow versions, modify line 15 in [Main.py]:
```python
# Change this:
from keras.utils.np_utils import to_categorical

# To this:
from keras.utils import to_categorical
```

---

## Project Structure

```
lomar-main/
├── Dataset/
│   └── mnist.csv            # MNIST handwritten digit dataset (28x28)
├── model/                   # Client-side trained models (generated at runtime)
│   ├── cnn_weights.keras    # Genuine model weights
│   └── poison_weights.keras # Poisoned model weights (label-flipping attack)
├── globalModel/
│   └── model.keras          # Global model accepted by the server
├── Main.py                  # Client GUI application (Tkinter)
├── Server.py                # Centralized federated learning server
├── run_headless.py          # Headless script for automated simulation
├── model_size_comparision.png    # Results: model compression comparison
├── no_defence_vs_lomar_accuracy.png  # Results: accuracy comparison
├── run.bat                  # Windows runner script
├── runServer.bat            # Windows server startup script
└── README.md
```

> **Note:** The `model/` and `globalModel/` folders are initially empty in a fresh clone. They are populated at runtime — `model/` stores trained client models and `globalModel/` stores the server's aggregated global model. The `.keras` weight files are generated during execution and are not version-controlled.

---

## How to Run

### 1. Start the Server
First, launch the centralized server which listens for incoming model updates:
```bash
python3 Server.py
```

### 2. Launch the Client GUI
In a separate terminal (with the virtual environment activated), start the client interface:
```bash
python3 Main.py
```
From the GUI, you can:
1. Load the MNIST dataset.
2. Preprocess the dataset.
3. Train and upload a genuine model.
4. Train and upload a poisoned model (simulating label-flipping attacks).
5. Compare accuracies and compression rates.

---

## Results

### 1. Defense Performance (No Defense vs LoMar)

The following chart compares the test accuracy of a model trained on poisoned data (label-flipping attack, no defense) against the genuine model used under the LoMar defense mechanism:

![No Defense vs LoMar Accuracy](no_defence_vs_lomar_accuracy.png)

- **No Defense (Poisoned Model):** The model trained on a label-flipped dataset and uploaded without defense.
- **LoMar Defense (Genuine Model):** The clean model accepted by the server after passing the LoMar anomaly detection check.

The server uses Kernel Density Estimation (KDE) on extracted features and accuracy thresholds to distinguish genuine updates from poisoned ones, rejecting malicious model updates while accepting legitimate ones.

### 2. Communication Efficiency — Model Size Compression

To reduce communication overhead in federated learning, model weights are compressed using `zlib` before transmission. The chart below compares the original and compressed model sizes:

![Model Size Comparison](model_size_comparision.png)

- **Original Model:** The raw `.keras` weights file before compression.
- **Compressed (zlib):** The same weights compressed using zlib, significantly reducing the payload size.

---
