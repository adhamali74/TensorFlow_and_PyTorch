import onnxruntime as ort
import numpy as np
from torchvision import datasets, transforms
import time

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)

# Convert dataset to numpy arrays
images = np.array([img.numpy() for img, _ in test_dataset])
labels = np.array([label for _, label in test_dataset])

# Reshape to match ONNX input
images = images.reshape(images.shape[0], 1, 28, 28).astype(np.float32)

# Load ONNX model
session = ort.InferenceSession("models/model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Run inference
print("Running inference on ONNX model...")
start_time = time.time()
outputs = session.run([output_name], {input_name: images})[0]
predictions = np.argmax(outputs, axis=1)
end_time = time.time()

# Calculate accuracy
accuracy = np.mean(predictions == labels) * 100
print(f"Inference accuracy: {accuracy:.4f}%")
print(f"Inference time: {end_time - start_time:.4f} seconds")
