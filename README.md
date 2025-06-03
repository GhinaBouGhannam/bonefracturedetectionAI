#  CNN-Based Fracture Detection from X-ray Images

##  Project Overview

This project develops a **Convolutional Neural Network (CNN)** model to automatically detect bone fractures in X-ray images. The goal is to assist radiologists and healthcare professionals with faster and more accurate diagnoses using machine learning.

- **Objective**: Classify X-ray images as *fractured* or *not fractured*.
- **Dataset**: Labeled X-ray images from [Kaggle](https://www.kaggle.com/), including rotated versions to improve model robustness. Here is the link for dataset [Bone Fracture Dataset](https://www.kaggle.com/datasets/devbatrax/fracture-detection-using-x-ray-images/data)  
- **Author**: Ghina Bou Ghannam

---

##  Key Benefits

- **Faster Diagnosis**  
  Helps healthcare professionals make quicker decisions.

- **Enhanced Accuracy**  
  Minimizes misdiagnosis with reliable detection.

- **Cost-Efficiency**  
  Reduces cost by automating manual review.

---

##  Dataset Preparation

- **Loading Images**: `cv2.imread`
- **Resizing**: `cv2.resize` ensures consistent input dimensions.
- **Storing Data**: Images are stored in NumPy arrays for efficient computation.
- **Normalization**: Pixel values scaled from `[0, 255]` to `[0, 1]`.

---

##  Dataset Splitting

- **Train/Test Split**: 80% training, 20% testing
- **Shuffling**: Applied to promote generalization

---

##  Image Augmentation

Uses `ImageDataGenerator` to:
- Create augmented image variations
- Fit and transform training data dynamically

---

##  CNN Model Architecture

- **Layers**:
  - Convolutional Layers
  - MaxPooling
  - Flatten
  - Dense (Fully Connected) Layers

- **Training**:
  - Batch Size: 32
  - Epochs: Up to 10
  - Early Stopping: Prevents overfitting

- **Compilation**:
  - Loss Function
  - Optimizer
  - Evaluation Metrics

---

##  Model Performance

### 1. Initial Phase
Gradual improvement in training accuracy; validation accuracy increases more slowly.

### 2. Stabilization Phase
Model learns robustly with consistent reduction in loss.

### 3. Mature Phase
Minimal generalization gap; model generalizes well and avoids overfitting.

---

##  Evaluation and Prediction

###  Image Prediction Workflow

1. **Preprocessing**: Load, resize, and normalize input images.
2. **Model Prediction**: Apply trained model to classify.
3. **Result Interpretation**: Output is either `Fractured` or `Not Fractured`.

---

