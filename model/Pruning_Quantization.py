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
