Here's a README file based on the provided document:

Automated Teeth Disease Detection Using Deep Learning
1. Introduction
This project focuses on automating the classification of teeth diseases from images using deep learning techniques, specifically Convolutional Neural Networks (CNNs). The motivation stems from the fact that manual classification by medical professionals is often time-consuming and prone to errors. Automated image classification, leveraging CNNs, can significantly assist medical professionals in achieving faster diagnoses.

2. Problem Statement
The primary goal of this project is to develop a CNN model capable of accurately classifying teeth diseases from images into one of seven distinct categories:

OC (Oral Cancer) 

CaS (Caries) 

OT (Other Types) 

CoS (Cosmetic Issues) 

Gum (Gum Disease) 

MC (Molar Conditions) 

OLP (Oral Lichen Planus) 

3. Dataset Description
The dataset used for this project is organized into three main subsets:



Training Set: Used to train the model.


Validation Set: Used for hyperparameter tuning and performance monitoring during training.


Testing Set: Used for the final evaluation of the model's performance on unseen data.

The dataset follows a specific hierarchy:

Teeth_Dataset/ [cite: 29]
├── Training/ [cite: 30]
├── Validation/ [cite: 30]
└── Testing/ [cite: 31]
Each of these directories contains subfolders named after the specific disease classes, and these subfolders hold the corresponding images.

4. Methodology
The project involves several key procedures and steps:

4.1. Data Loading and Exploration
The dataset is loaded from its directory structure, with labels automatically assigned based on folder names. Initial exploration includes visualizing the distribution of classes and plotting random images from each disease category across the training, validation, and test sets.





4.2. Preprocessing and Augmentation
A Load_and_preprocessing() function prepares the training and validation datasets.


Image Augmentation (Training Set Only): To enhance model generalization and prevent overfitting, training images undergo random transformations including rescaling, rotation, width/height shifts, shearing, zooming, horizontal flipping, and filling of missing pixels using the nearest strategy.




Image Normalization (Validation Set): Only rescaling (pixel values normalized to [0,1] range) is applied to the validation set to ensure accurate model evaluation without augmentation.



Data Generator: Both sets are passed through flow_from_directory() to automatically assign labels, resize images (e.g., to 150x150), and return batches of image-label pairs using one-hot encoding.



4.3. CNN Model Architecture
The CNN model is designed to classify input dental images into 7 disease categories using a sequential architecture of convolutional, pooling, and dense layers.

The layer-by-layer breakdown is as follows:


Conv2D (32 filters, 3x3 kernel, ReLU activation): Extracts local features.


MaxPooling2D (2x2 pool size): Downsamples feature maps, reducing dimensionality.


Conv2D (64 filters, 3x3 kernel, ReLU activation): Learns more complex patterns.


MaxPooling2D (2x2 pool size): Further reduces spatial dimensions.


Conv2D (128 filters, 3x3 kernel, ReLU activation): Learns high-level features for fine-grained distinction.


MaxPooling2D (2x2 pool size): Prepares for dense layers.


Flatten: Converts 2D output to a 1D vector.


Dense (512 units, ReLU activation): A fully connected decision-making layer.


Dropout (rate=0.5): Randomly drops neurons to prevent overfitting.


Dense (7 units, Softmax activation): Final output layer providing probability distribution over 7 classes.

4.4. Training Configuration
The model is compiled with the following settings:



Loss Function: Categorical Crossentropy 



Optimizer: Adam (with a learning rate of 0.001) 



Metrics: Accuracy 


Callbacks:


EarlyStopping: Monitors val_loss, with a patience of 4 epochs, and restores best weights.




ReduceLROnPlateau: Monitors val_loss, reduces learning rate by a factor of 0.2, with a patience of 5 epochs, and a minimum learning rate of 0.00001.


5. Train and Evaluation
After training for 100 epochs, the model achieved the following performance metrics:


Training Accuracy: 0.9112 (loss: 0.2765) 


Validation Accuracy: 0.9786 (loss: 0.0863) 


Test Accuracy: 0.9838 (loss: 0.0679) 

Further evaluation was performed using a Classification Report and Confusion Matrix generated with 

sklearn.metrics.

6. Conclusion
This project successfully developed a deep learning pipeline for classifying teeth diseases, demonstrating decent accuracy. The CNN model effectively learned to differentiate between 7 types of diseases using real-world image data. With further tuning, this model has the potential to support clinical diagnosis as a valuable decision support tool.


Author: Ahmed Mostafa Gamal Eddin, Computer Vision Intern at Cellula Technologies 



Date: July 2, 2025 
