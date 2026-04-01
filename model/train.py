import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import joblib

data_dir = "../data/train"

X = []
y = []

labels = os.listdir(data_dir)

for label in labels:
    path = os.path.join(data_dir, label)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48,48))
        X.append(img.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# =====================
# Model 1: ML
# =====================
model_ml = RandomForestClassifier(n_estimators=50)
model_ml.fit(X, y)
joblib.dump(model_ml, "model_ml.pkl")

# =====================
# Model 2: Neural Network
# =====================
model_nn = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=200)
model_nn.fit(X, y)
joblib.dump(model_nn, "model_nn.pkl")

print("Train เสร็จแล้ว")