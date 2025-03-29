## 1. Imports & Utilities **

import os
import math
import zipfile
import numpy as np
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers, backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_curve, auc
)

def get_zipped_model_size(file):
    zipped_file = file + '.zip'
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)
    return os.path.getsize(zipped_file)

## 2. Residual Attention Block & ECA Layers

import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Flatten, GRU, GlobalAveragePooling1D,
    Activation, Reshape, Multiply, Permute, Lambda
)

# Residual Attention Block
def residual_attention_block(x, units):
    a = Permute((2, 1))(x)
    a = Dense(K.int_shape(x)[1], activation='relu')(a)  
    a = Permute((2, 1))(a)
    x = Multiply()([x, a])
    return x

# ECA Block 
def eca_layer(inputs_tensor):
    channels = K.int_shape(inputs_tensor)[-1]
    x = GlobalAveragePooling1D()(inputs_tensor)
    x = Reshape((channels, 1))(x)
    x = Conv1D(1, kernel_size=1, padding="same", activation='relu')(x)  # kernel size=1, activation=ReLU
    x = Activation('sigmoid')(x)
    x = Reshape((1, channels))(x)
    return Multiply()([inputs_tensor, x])

## 3. Build & Train the Model

import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Dense, Flatten, GRU, GlobalAveragePooling1D,
    Activation, Reshape, Multiply, Permute, Lambda
)

input_layer = Input(shape=(n_timesteps, n_features))

# 1D CNN + ReLU
x = Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)

# Residual Attention
x = residual_attention_block(x, units=64)

# GRU 
x = GRU(64, return_sequences=True, activation='relu')(x)

# ECA Attention
x = eca_layer(x)

# Flatten + Dense
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

# Build model
model = Model(inputs=input_layer, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

## 4. Training & Evaluation Visualization

# Loss Plot
plt.plot(history.history['loss'], 'y', label='Train Loss')
plt.plot(history.history['val_loss'], 'r', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Accuracy Plot
plt.plot(history.history['accuracy'], 'y', label='Train Acc')
plt.plot(history.history['val_accuracy'], 'r', label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## 5. Evaluation Metrics

predy = model.predict(X_test)
predy = np.argmax(predy, axis=-1)
LABELS = ['FOG', 'Non-FOG']

cm = confusion_matrix(y_test, predy)
print(cm)
print("Accuracy:", accuracy_score(y_test, predy))
print("F1 Score:", f1_score(y_test, predy, average='weighted'))
print(classification_report(y_test, predy))

# Confusion Matrix Plot
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Sensitivity & Specificity
def calculate_sensitivity_specificity(y_test, predy):
    tn, fp, fn, tp = confusion_matrix(y_test, predy).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity

sensitivity, specificity = calculate_sensitivity_specificity(y_test, predy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, predy)
auc_score = auc(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=f'(AUC = {auc_score:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.grid(True)
plt.title('ROC Curve')
plt.show()

