** ## 1. Imports & Utilities **

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

## 2. Custom Attention & ECA Layers

def attention_3d_block(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
        
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def eca_layer(inputs_tensor=None, num=None, gamma=10, b=1):
    channels = K.int_shape(inputs_tensor)[-1]
    t = int(abs((math.log(channels, 2) + b) / gamma))
    k = t if t % 2 else t + 1

    x = GlobalAveragePooling1D()(inputs_tensor)
    x = Reshape((channels, 1))(x)
    x = Conv1D(1, kernel_size=k, padding="same", name="eca_conv1_" + str(num))(x)
    x = Activation('sigmoid', name='eca_conv1_relu_' + str(num))(x)
    x = Reshape((1, 1, channels))(x)
    
    return multiply([inputs_tensor, x])


## 3. Build & Train the Model

input_layer = Input(shape=(n_timesteps, n_features))
x = Conv1D(filters=64, kernel_size=1, activation='relu')(input_layer)
x = attention_3d_block(x)
x = layers.GRU(64, return_sequences=True, activation="relu")(x)

x = eca_layer(x)

x = Flatten()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(2, activation="sigmoid")(x)

model = Model(input_layer, outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, trainy_one_hot, epochs=20, batch_size=32, validation_split=0.1, shuffle=True)


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

## 6. Save, Quantize & Evaluate TFLite Model

mfile = 'model.h5'
model.save(mfile, include_optimizer=False)
model_size = get_zipped_model_size(mfile)
print("Model Size: {:.2f}MB".format(model_size / 1000))

# Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
quantized_model = converter.convert()

tflite_file = 'tflite_model'
with open(tflite_file, 'wb') as f:
    f.write(quantized_model)

quantized_model_size = get_zipped_model_size(tflite_file)
print("Quantized Model Size: {:.2f}MB".format(quantized_model_size / 1000))

interpreter_quant = tf.lite.Interpreter(model_path=tflite_file)
interpreter_quant.allocate_tensors()


## 7. Inference on TFLite Model

def evaluate_model(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_idx = input_details[0]['index']
    output_idx = output_details[0]['index']

    predictions = []
    for i, test_data in enumerate(X_test):
        test_data = np.expand_dims(test_data, axis=0).astype(np.float32)
        interpreter.set_tensor(input_idx, test_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_idx)
        digit = np.argmax(output, axis=1)
        predictions.append(digit)

    predictions = np.array(predictions)
    labels = y_test.reshape(-1, 1)

    accuracy = np.sum(predictions == labels) / len(predictions) * 100
    return accuracy

tflite_accuracy = evaluate_model(interpreter_quant)
print("TFLite Accuracy: {:.2f}%".format(tflite_accuracy))

# 8. Pruning + Quantization

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
num_samples = X_train.shape[0] * (1 - 0.1)
end_step = np.ceil(num_samples / 32).astype(np.int32) * 5

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,
        final_sparsity=0.8,
        begin_step=0,
        end_step=end_step
    )
}

pruned_model = prune_low_magnitude(model, **pruning_params)
pruned_model.compile(optimizer='adam',
                     loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=['accuracy'])

logdir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
]

pruned_model.fit(X_train, trainy_one_hot, batch_size=32, epochs=5, validation_split=0.1, callbacks=callbacks)

# Strip and save
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
pruned_file = 'pruned_model.h5'
pruned_model.save(pruned_file, include_optimizer=False)
pruned_model_size = get_zipped_model_size(pruned_file)
print("Pruned Model Size: {:.2f}MB".format(pruned_model_size / 1000))

# Quantize the pruned model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
quantized_pruned_model = converter.convert()

quantized_pruned_file = 'quantized_pruned_model.tflite'
with open(quantized_pruned_file, 'wb') as f:
    f.write(quantized_pruned_model)

quantized_pruned_model_size = get_zipped_model_size(quantized_pruned_file)
print("Quantized Pruned Model Size: {:.2f}MB".format(quantized_pruned_model_size / 1000))

interpreter_pq = tf.lite.Interpreter(model_path=quantized_pruned_file)
interpreter_pq.allocate_tensors()
quantized_pruned_model_accuracy = evaluate_model(interpreter_pq)
print("Quantized Pruned Accuracy: {:.2f}%".format(quantized_pruned_model_accuracy))

# 9. Final Model Size & Accuracy Comparison

# Size Comparison
sizes = [model_size/1000, quantized_model_size/1000, pruned_model_size/1000, quantized_pruned_model_size/1000]
labels = ['Original', 'Quantized', 'Pruned', 'Quantized + Pruned']

plt.bar(labels, sizes)
plt.title('Model Size Comparison')
plt.ylabel('Size (MB)')
plt.show()

# Accuracy Comparison
accuracies = [base_accuracy*100, tflite_accuracy, model_for_pruning_accuracy*100, quantized_pruned_model_accuracy]
plt.bar(labels, accuracies)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.show()



