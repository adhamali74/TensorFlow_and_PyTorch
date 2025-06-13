import tensorflow as tf
import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# === Step 1: Load and preprocess MNIST ===
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0  # Normalize to [0,1]
x_test = x_test.astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# === Step 2: Build the model ===
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Flatten

inputs = Input(shape=(28, 28))
x = Flatten()(inputs)
x = Dense(64, activation='relu')(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)



model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Step 3: Train the model and time it ===
print("Training TensorFlow model...")
start_time = time.time()
history = model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=2)
training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")

# === Step 4: Evaluate model ===
print("Evaluating TensorFlow model...")
eval_start = time.time()
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
eval_time = time.time() - eval_start
print(f"Test accuracy: {test_acc:.4f}")
print(f"Inference time on test set: {eval_time:.4f} seconds")

# === Step 5: Export to TensorFlow Lite ===
print("Exporting model to TFLite...")
# Define concrete function from model
input_spec = tf.TensorSpec([None, 28, 28], tf.float32)
concrete_func = tf.function(model).get_concrete_function(input_spec)

# Convert using concrete function
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

with open("models/model.tflite", "wb") as f:
    f.write(tflite_model)
print("Exported TFLite model saved to models/model.tflite")
