# 🧠 NeuroDrive: End-to-End Autonomous Driving using Deep Learning

A deep learning-based autonomous driving system that predicts vehicle steering angles directly from raw camera images using a custom Convolutional Neural Network (CNN). Trained and tested on the [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim), this project demonstrates real-time, end-to-end vehicle control powered by computer vision and regression modeling.

---

## 🚀 Overview

**NeuroDrive** simulates human-like driving behavior using a deep CNN trained on image data captured from a virtual driving simulator. Instead of relying on rule-based systems or lane detection algorithms, this project learns to map camera images directly to steering commands — enabling a fully end-to-end solution.

**Key Features:**
- Custom CNN architecture (inspired by NVIDIA’s PilotNet)
- Real-time inference via Flask + SocketIO
- Image preprocessing pipeline (crop, resize, blur, normalize)
- Data augmentation (zoom, pan, brightness, flipping)
- Balanced training using steering angle histogram analysis

---

## 🧠 Model Architecture

The architecture follows a CNN-based regression approach:

- 4 Convolutional Layers with ELU activation
- Stride-based downsampling `(2,2)`
- Fully Connected Layers
- Output Layer: Single neuron for continuous steering angle prediction

**Loss Function:** `Mean Squared Error (MSE)`  
**Optimizer:** `Adam` (learning rate = `0.001`)  
**Activation:** `ELU` for smoother learning and avoiding dead neurons

---

## 📊 Dataset

Data was captured using the simulator by manually driving the car:

- Each row contains: `center`, `left`, `right` image paths and a `steering angle`
- Images captured at 3 angles simulate multi-view perception
- Dataset imbalance was mitigated using binning and undersampling

---

## 🔄 Data Preprocessing & Augmentation

### ✅ Preprocessing:
- **Crop:** Remove sky and dashboard from image: `img = img[60:135,:,:]`
- **Color Space:** Convert RGB to YUV (used in NVIDIA architecture)
- **Blur:** Gaussian Blur to reduce noise
- **Resize:** Downscale to `(200, 66)`
- **Normalize:** Scale pixel values to `[0, 1]`

### 🌀 Augmentation Techniques:
- **Zoom:** Simulates forward movement
- **Pan:** Random translation to simulate car position shift
- **Brightness:** Varies lighting conditions
- **Flip:** Horizontally flip images and invert steering angles

---

## 🧮 Training Details

| Parameter          | Value        |
|--------------------|--------------|
| Batch Size         | 100          |
| Epochs             | 10–20        |
| Steps per Epoch    | 300          |
| Validation Steps   | 200          |
| Validation Split   | 20%          |

A Python `generator` was used for memory-efficient loading and real-time augmentation during training.

---

## 🖥️ Real-Time Inference

The `drive.py` file implements a real-time server using Flask and SocketIO that:
1. Receives camera input from the simulator (base64-encoded)
2. Preprocesses the input
3. Predicts the steering angle using the trained model
4. Calculates throttle based on speed
5. Sends control signals back to the simulator

