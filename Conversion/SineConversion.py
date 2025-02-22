import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#  Step 1: Generate Sine Wave Training Data
num_train_examples = 20000
sequence_length = 8
batch_size = 64
num_epochs = 50
val_split = 0.2
rng = np.random.default_rng(2024)

# Generate sine waves with random frequencies and phase shifts
frequencies = rng.uniform(0.02, 0.2, size=num_train_examples)
phase_offsets = rng.uniform(0.0, 2*np.pi, size=num_train_examples)
sequences = np.zeros((num_train_examples, sequence_length))

for i in range(num_train_examples):
    sequences[i] = np.sin(2 * np.pi * frequencies[i] * np.arange(sequence_length) + phase_offsets[i])

# Split into input (first 7 values) and target (8th value)
x_train = sequences[:, :sequence_length-1]
y_train = sequences[:, sequence_length-1]

# Normalize inputs to range [-1,1] (important for TFLM!)
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)

#  Step 2: Build a Fully Connected Model (LSTM removed)
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1)  # Single neuron for regression
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

#  Step 3: Train the Model
train_hist = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=val_split)

#  Step 4: Plot Training Loss
plt.plot(train_hist.epoch, train_hist.history['loss'], 'b-', label='Train Loss')
plt.plot(train_hist.epoch, train_hist.history['val_loss'], 'r-', label='Val Loss')
plt.legend()
plt.grid(True)
plt.show()

# Step 5: Evaluate and Test Predictions
loss = model.evaluate(x_train, y_train)
print("Final training loss:", loss)

# Predict on training data
y_pred = model.predict(x_train)
plt.scatter(y_train, y_pred, alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.title("Predicted vs. Actual Outputs")
plt.grid(True)
plt.show()

#  Step 6: Save the Model for Further Conversion
model.save("sin_predictor.h5")

#  Step 7: Convert to TensorFlow Lite for TFLM (with Int8 Quantization)
def representative_dataset():
    for i in range(100):
        yield [x_train[i].reshape(1, sequence_length-1)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save the model
with open("sin_predictor_int8.tflite", "wb") as f:
    f.write(tflite_model)

print(" TFLite model successfully saved as sin_predictor_int8.tflite!")

#  Step 8: Convert `.tflite` to `.h` for Arduino
TFLITE_MODEL_PATH = "sin_predictor_int8.tflite"
HEADER_FILENAME = "sin_predictor_model.h"

# Read the TFLite model file
with open(TFLITE_MODEL_PATH, "rb") as f:
    tflite_model = f.read()

# Convert model data to a C byte array format
c_array = ", ".join(f"0x{b:02x}" for b in tflite_model)

# Generate the header file content
header_content = f"""#ifndef SIN_PREDICTOR_MODEL_H
#define SIN_PREDICTOR_MODEL_H

#include <stddef.h>
#include <stdint.h>

// Align model in memory for TFLM compatibility
alignas(8) const unsigned char sin_predictor_int8_tflite[] = {{
    {c_array}
}};

const size_t sin_predictor_int8_tflite_len = {len(tflite_model)};

#endif // SIN_PREDICTOR_MODEL_H
"""

# Write the header file
with open(HEADER_FILENAME, "w") as f:
    f.write(header_content)

print(f" Successfully generated {HEADER_FILENAME}!")
