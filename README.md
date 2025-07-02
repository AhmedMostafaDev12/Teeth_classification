# Automated Teeth Disease Detection Using Deep Learning

## 1. Introduction

This project focuses on automating the classification of teeth diseases from images using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The motivation stems from the fact that manual classification by medical professionals is often time-consuming and prone to errors. Automated image classification, leveraging CNNs, can significantly assist medical professionals in achieving faster diagnoses.

## 2. Problem Statement

The primary goal of this project is to develop a CNN model capable of accurately classifying teeth diseases from images into one of seven distinct categories:
* OC (Oral Cancer)
* CaS (Caries)
* OT (Other Types)
* CoS (Cosmetic Issues)
* Gum (Gum Disease)
* MC (Molar Conditions)
* OLP (Oral Lichen Planus)

## 3. Dataset Description

The dataset used for this project is organized into three main subsets:
* **Training Set:** Used to train the model.
* **Validation Set:** Used for hyperparameter tuning and performance monitoring during training.
* **Testing Set:** Used for the final evaluation of the model's performance on unseen data.

The dataset follows a specific hierarchy:
Teeth_Dataset/
├── Training/
├── Validation/
└── Testing/
Each of these directories contains subfolders named after the specific disease classes, and these subfolders hold the corresponding images. The loading process reads these directories and automatically assigns labels based on folder names, ensuring that the data is structured appropriately for supervised learning.

## 4. Methodology

The project involves several key procedures and steps:

### 4.1. Data Loading and Exploration

The dataset is loaded from its directory structure using `tf.keras.utils.image_dataset_from_directory`, with labels automatically assigned based on folder names. Initial exploration includes visualizing the distribution of classes and plotting random images for each class across the validation, training, and test sets.

### 4.2. Preprocessing and Augmentation

A `Load_and_preprocessing()` function is responsible for preparing the training and validation datasets before feeding them into the Convolutional Neural Network (CNN).

* **Image Augmentation (Training Set Only):** To improve the generalization of the model and prevent overfitting, the training images undergo a series of random transformations:
    * Rescaling: Pixel values are normalized from the range [0,255] to [0, 1].
    * Random rotation
    * Width and height shifts
    * Shearing transformation
    * Zooming in and out
    * Horizontal flipping
    * Filling of missing pixels using the nearest strategy
* **Image Normalization (Validation Set):** For the validation set, only rescaling (pixel values normalized to the [0, 1] range) is applied. No augmentation is applied to validation images to ensure an accurate evaluation of model performance.
* **Data Generator:** Both training and validation sets are passed through the `flow_from_directory()` method, which automatically assigns labels based on folder names, resizes images to a specified target size (e.g., 150x150), and returns batches of image-label pairs using one-hot encoding (categorical class mode).

### 4.3. CNN Model Architecture

The Convolutional Neural Network (CNN) model used in this project is designed to classify input dental images into 7 disease categories. It follows a sequential architecture consisting of multiple convolutional and pooling layers, followed by dense layers for classification.

The layer-by-layer breakdown is as follows:
* **Conv2D (32 filters, 3x3 kernel, ReLU activation):** Extracts 32 local features from the input image using small 3x3 filters with non-linearity introduced via ReLU.
* **MaxPooling2D (2x2 pool size):** Downsamples feature maps by taking the maximum value in each 2x2 region, reducing dimensionality and computation.
* **Conv2D (64 filters, 3x3 kernel, ReLU activation):** Increases the depth of the network to learn more complex patterns and features.
* **MaxPooling2D (2x2 pool size):** Further reduces the spatial dimensions to retain only the most prominent features.
* **Conv2D (128 filters, 3x3 kernel, ReLU activation):** Adds more abstraction by learning high-level features, useful for distinguishing fine-grained disease types.
* **MaxPooling2D (2x2 pool size):** Reduces spatial dimensions again to prepare for the dense layers.
* **Flatten:** Converts the 2D output of the last pooling layer into a 1D vector for input into the fully connected layers.
* **Dense (512 units, ReLU activation):** A fully connected layer acting as the decision-making layer with ReLU activation to introduce non-linearity.
* **Dropout (rate=0.5):** Randomly drops 50% of neurons during training to prevent overfitting and improve generalization.
* **Dense (7 units, Softmax activation):** Final output layer that produces a probability distribution over the 7 disease classes for classification.

### 4.4. Training Configuration

The model is compiled with the following settings:
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam (with a learning rate of 0.001)
* **Metrics:** Accuracy
* **Callbacks:**
    * **EarlyStopping:** Monitors `val_loss`, with a patience of 4 epochs, and restores best weights.
    * **ReduceLROnPlateau:** Monitors `val_loss`, reduces learning rate by a factor of 0.2, with a patience of 5 epochs, and a minimum learning rate of 0.00001.

## 5. Train and Evaluation

After training for 100 epochs, the model achieved the following performance metrics:
* **Training Accuracy:** accuracy: 0.9112, loss: 0.2765
* **Validation Accuracy:** accuracy: 0.9786, loss: 0.0863
* **Test Accuracy:** accuracy: 0.9838, loss: 0.0679

Further evaluation was performed using a Classification Report and Confusion Matrix generated with `sklearn.metrics`. Visualizations of Model Accuracy and Model Loss over epochs are also provided.

## 6. Conclusion

In this project, a deep learning pipeline was successfully built for classifying teeth diseases, demonstrating decent accuracy. The CNN model effectively learned to differentiate between 7 types of diseases using real-world image data. With further tuning, this model could potentially support clinical diagnosis as a valuable decision support tool.

---

**Author:** Ahmed Mostafa Gamal Eddin
**Role:** Computer Vision Intern
**Affiliation:** Cellula Technologies
**Date:** July 2, 2025
