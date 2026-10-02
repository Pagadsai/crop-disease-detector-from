# Crop Disease Detector
# 🌱 CropGuard — Crop Disease Detector

CropGuard is an AI-powered web application that analyzes crop leaf
images and identifies potential plant diseases.

The application uses a MobileNet-based TensorFlow Lite model trained
on PlantVillage-style crop disease classes.

---

## 🚀 Features

- Upload crop leaf images
- Drag-and-drop image upload
- Tomato, Potato and Bell Pepper support
- AI-based disease classification
- Confidence score
- Treatment and prevention recommendations
- Crop mismatch detection
- Prediction history
- Responsive web interface
- Works on desktop and mobile browsers
- Local browser-based prediction history

---

## 🌾 Supported Crops

### Tomato

- Bacterial Spot
- Early Blight
- Healthy
- Late Blight
- Leaf Mold
- Mosaic Virus
- Septoria Leaf Spot
- Yellow Leaf Curl Virus

### Potato

- Early Blight
- Healthy
- Late Blight

### Bell Pepper

- Healthy
- Leaf Spot

---

## 🧠 Machine Learning

The application uses a MobileNet-based TensorFlow Lite model.

### Image preprocessing

Images are:

1. Converted to RGB
2. Resized to 224 × 224 pixels
3. Normalized to the `[0,1]` range
4. Standardized using ImageNet mean and standard deviation

The model produces predictions for 13 classes.

The application then applies crop-specific validation to ensure that
the selected crop matches the model's predicted crop.

---

## 🏗️ Architecture

```text
                 User
                   |
                   ↓
            Web Interface
                   |
                   ↓
            Image Upload
                   |
                   ↓
          FastAPI Backend
                   |
                   ↓
       Image Preprocessing
                   |
                   ↓
       MobileNet TFLite Model
                   |
                   ↓
          Disease Prediction
                   |
          ┌────────┴────────┐
          ↓                 ↓
     Confidence        Treatment
                         Advice
          |
          ↓
    Prediction History
