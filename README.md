# Lab03: TensorFlow and PyTorch

**Author:** Adham Ali Mohamed Abdelfattah  
**Student ID:** 22404698  
**Course:** Advanced Programming  
**Instructor:** Prof. Tobias Schaffer  
**Date:** June 13, 2025

---

## 1. Introduction

This lab compares two popular deep learning frameworks: TensorFlow and PyTorch. The task was to build, train, evaluate, and export a simple feedforward neural network using both libraries on the MNIST dataset. Exported models were tested with TensorFlow Lite (TFLite) and ONNX Runtime to assess portability and inference performance.

---

## 2. Methodology

### 2.1 Software and Hardware Used

- **Programming Language:** Python 3.12  
- **Libraries:** TensorFlow 2.16.2, PyTorch 2.2.2, ONNX, ONNXRuntime, TorchVision, Matplotlib  
- **Hardware:** Apple MacBook Air (Intel Core i5, CPU only)

### 2.2 Code Repository Structure


```
AP_Project/
├── src/                # All source code for training, exporting, inference
│   ├── tensorflow_model.py
│   ├── pytorch_model.py
│   └── onnx_inference.py
├── models/             # Saved models (.tflite, .onnx)
├── logs/               # Log files for training and inference
└── README.md           # This file
```

### 2.3 Code Implementation

- TensorFlow: `Sequential` model using `Flatten`, `Dense`, and `ReLU` layers.
- PyTorch: Custom `nn.Module` class with `Linear` and `ReLU` layers.
- ONNX export from PyTorch model.
- TFLite export from TensorFlow model.

---

## 3. Results

| Framework    | Test Accuracy | Inference Time |
|--------------|----------------|----------------|
| TensorFlow   | 97.14%         | 1.36 sec       |
| PyTorch      | 97.39%         | 5.19 sec       |
| ONNXRuntime  | 97.39%         | **0.045 sec**  |

---

## 4. Challenges, Limitations, and Error Analysis

### 4.1 Challenges Faced

- TFLite export required a workaround for input signature specification.
- ONNX export in PyTorch required input formatting in `NCHW` layout.

### 4.2 Error Analysis

- TensorFlow threw conversion errors related to `_get_save_spec`.
- ONNXRuntime required consistent input shapes for inference.

### 4.3 Limitations

- Training on CPU (Intel Core i5) limited performance.
- Only basic fully connected networks were used; no convolutional models were explored.

---

## 5. Discussion

TensorFlow and PyTorch both provided high accuracy on the MNIST task. However, PyTorch had a smoother export to ONNX. ONNXRuntime drastically reduced inference time, making it ideal for deployment. TensorFlow’s TFLite required extra setup but worked after format adjustments.

---

## 6. Conclusion

Both frameworks are viable for developing simple deep learning models. PyTorch with ONNXRuntime yielded the best inference speed and smoother export process. The experiment highlights the importance of interoperability and performance tuning in real-world deployments.

---

## 7. References

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [ONNX](https://onnx.ai/)
