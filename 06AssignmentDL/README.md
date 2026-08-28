# CNN-Based Tomato Leaf Disease Classification

## 1. Project Overview

This project implements a **Convolutional Neural Network (CNN)** for classifying tomato leaf images into 10 disease categories.
The model learns visual features such as leaf spots, color patterns, textures and shapes from tomato leaf images and predicts the corresponding disease.
The project is implemented in **Python using TensorFlow, Keras and Keras Tuner** in Google Colab.
### Main Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Normalization & Augmentation
   ↓
CNN Model
   ↓
Training
   ↓
Evaluation
   ↓
Hyperparameter Tuning
   ↓
Final CNN
   ↓
Disease Prediction
```

---

## 2. Problem Statement

**Design and implement a Convolutional Neural Network (CNN) for image classification using the Tomato or Soybean disease dataset.**

For this assignment, the **Tomato Leaf Disease Dataset** was selected.

The system classifies a tomato leaf image into its corresponding disease category.

---

## 3. Objectives

* Understand CNN-based image classification.
* Load and explore a real-world image dataset.
* Resize and normalize images.
* Apply data augmentation.
* Build and train a CNN.
* Evaluate the model using standard metrics.
* Tune hyperparameters using Keras Tuner.
* Build the final tuned CNN.
* Save the trained model.
* Predict the disease of a new tomato leaf image.

---

# 4. Dataset

### Dataset

**Tomato Leaf Disease Dataset**

### Source

Kaggle:
https://www.kaggle.com/datasets/kaustubhb999/tomatoleaf

### Dataset Details

| Property                | Value  |
| ----------------------- | ------ |
| Number of Classes       | 10     |
| Training Images         | 10,000 |
| Validation Images       | 1,000  |
| Training Images/Class   | 1,000  |
| Validation Images/Class | 100    |

### Classes

1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites
7. Tomato Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Tomato Healthy

The dataset provides separate **training** and **validation** folders.

> A separate test folder is not provided, so the validation dataset was used for final evaluation.

---

# 5. What is CNN?

A **Convolutional Neural Network (CNN)** is a deep learning model mainly used for image classification and computer vision.

CNNs automatically learn useful features from images instead of requiring manual feature extraction.

For tomato leaves, the CNN can learn:

* Edges
* Shapes
* Textures
* Spots
* Color patterns
* Disease-related features

### General CNN Architecture

```text
Input Image
     ↓
Convolution
     ↓
ReLU
     ↓
Max Pooling
     ↓
Convolution
     ↓
Max Pooling
     ↓
Flatten
     ↓
Dense
     ↓
Dropout
     ↓
Softmax
     ↓
Prediction
```

---

# 6. CNN Used in This Project

The images are resized to **128 × 128 × 3**, where 3 represents the RGB channels.

The final CNN contains:

```text
Input: 128 × 128 × 3
        ↓
Data Augmentation
        ↓
Conv2D - 32 filters
        ↓
MaxPooling
        ↓
Conv2D - 128 filters
        ↓
MaxPooling
        ↓
Conv2D - 128 filters
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense - 64 neurons
        ↓
Dropout - 0.3
        ↓
Dense - 10 neurons
        ↓
Softmax
```

---

# 7. CNN Layers

### Conv2D

`Conv2D` applies filters to the image and extracts features such as edges, textures and patterns.

### ReLU

ReLU introduces non-linearity and helps the network learn complex relationships.

### MaxPooling2D

Max pooling reduces the spatial size of feature maps while retaining important information.

### Flatten

Flatten converts the feature maps into a one-dimensional vector.

### Dense

Dense layers use the extracted features for classification.

### Dropout

Dropout randomly disables neurons during training and helps reduce overfitting.

### Softmax

The final softmax layer produces probabilities for all 10 disease classes. The class with the highest probability is selected as the prediction.

---

# 8. Image Preprocessing

Image preprocessing is performed before training.

### Image Resizing

All images are resized to:

**128 × 128 pixels**

This ensures that every input image has the same dimensions.

Implemented using:

```python
tf.keras.utils.image_dataset_from_directory()
```

with the `image_size` parameter.

### Pixel Normalization

Image pixel values originally range from **0 to 255**.

They are converted to **0 to 1** using:

```python
tf.keras.layers.Rescaling(1./255)
```

Normalization helps make neural network training more stable.

---

# 9. Data Augmentation

Data augmentation creates variations of training images.

The following operations were used:

```python
RandomFlip("horizontal")
RandomRotation(0.1)
RandomZoom(0.1)
```

### Purpose

* Reduce overfitting
* Improve generalization
* Make the model more robust
* Handle variations in leaf orientation and size

The augmentation layer is included in the CNN and is applied during training.

---

# 10. Dataset Loading

The dataset was loaded using:

```python
tf.keras.utils.image_dataset_from_directory()
```

This function automatically:

* Reads images from folders.
* Assigns class labels.
* Resizes images.
* Creates TensorFlow datasets.
* Creates batches.

The batch size used was **32**.

---

# 11. Model Compilation

The CNN was compiled using:

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

### Adam Optimizer

Adam updates the model weights during training based on calculated gradients.

### Sparse Categorical Crossentropy

This loss function is suitable because the classes are represented using integer labels.

### Accuracy

Accuracy measures the percentage of correctly classified images.

---

# 12. Model Training

The model was trained using:

```python
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)
```

During training, the model learns the relationship between tomato leaf images and their disease classes.

The following values were monitored:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss

Training and validation graphs were plotted to observe model performance.

---

# 13. Model Evaluation

The model was evaluated using:

```python
model.evaluate(val_ds)
```

The evaluation provides the loss and accuracy of the model on the validation data.

Additional evaluation was performed using:

* Confusion Matrix
* Precision
* Recall
* F1-score
* Classification Report

---

# 14. Hyperparameter Tuning

**Keras Tuner** was used to improve the CNN configuration.

The tuning algorithm used was:

```python
kt.RandomSearch()
```

Random Search tests different combinations of hyperparameters and selects a configuration based on validation accuracy.

### Hyperparameters Tuned

| Hyperparameter        | Values        |
| --------------------- | ------------- |
| First Conv2D Filters  | 32, 64        |
| Second Conv2D Filters | 64, 128       |
| Dense Units           | 64, 128       |
| Dropout               | 0.3, 0.5      |
| Learning Rate         | 0.001, 0.0005 |

A total of **5 trials** were performed with **3 epochs per trial**.

---

# 15. Best Hyperparameters

Keras Tuner selected the following configuration:

| Parameter     | Best Value |
| ------------- | ---------- |
| Filters 1     | 32         |
| Filters 2     | 128        |
| Dense Units   | 64         |
| Dropout       | 0.3        |
| Learning Rate | 0.001      |

These values were used to build the final CNN.

---

# 16. Final CNN

The final CNN uses the best hyperparameters obtained from Keras Tuner.

The final architecture contains:

* 3 convolution layers
* 3 max pooling layers
* 1 flatten layer
* 1 dense layer
* 1 dropout layer
* 1 softmax output layer

The model contains approximately **1.79 million trainable parameters**.

---

# 17. Confusion Matrix

A **confusion matrix** is used to analyze the performance of a classification model.

It compares the:

* Actual class
* Predicted class

Since this project contains 10 classes, the confusion matrix contains **10 × 10 entries**.

### Interpretation

* Diagonal values → Correct predictions
* Non-diagonal values → Incorrect predictions

The confusion matrix helps identify which tomato diseases are correctly classified and which diseases are commonly confused with one another.

It was generated using:

```python
confusion_matrix(y_true, y_pred)
```

and visualized using a Seaborn heatmap.

---

# 18. Classification Report

The classification report provides class-wise performance metrics.

It contains:

### Precision

Measures how many samples predicted as a particular class actually belong to that class.

### Recall

Measures how many actual samples of a class were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### Support

Number of actual samples belonging to each class.

The report was generated using:

```python
classification_report(
    y_true,
    y_pred,
    target_names=class_names
)
```

This provides more detailed information than accuracy alone.

---

# 19. Prediction on New Images

The trained model can classify a new tomato leaf image.

### Prediction Process

```text
New Image
   ↓
Resize to 128 × 128
   ↓
Convert to Array
   ↓
Normalize
   ↓
CNN
   ↓
Softmax
   ↓
Highest Probability
   ↓
Predicted Disease
```

The predicted class is obtained using:

```python
np.argmax(prediction[0])
```

The highest softmax probability is used as the confidence score.

Example:

```text
Predicted Disease: Tomato___Early_blight
Confidence: 99.98%
```

The confidence value is specific to that individual prediction and should not be considered the overall model accuracy.


---

# 20. Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| TensorFlow   | Deep learning framework        |
| Keras        | CNN model development          |
| Keras Tuner  | Hyperparameter tuning          |
| NumPy        | Numerical operations           |
| Scikit-learn | Model evaluation               |
| Matplotlib   | Graph visualization            |
| Seaborn      | Confusion matrix visualization |
| Google Colab | Development environment        |
| Kaggle       | Dataset source                 |

---

# 21. Important Python and ML Functions

| Function / Operation             | Purpose                    |
| -------------------------------- | -------------------------- |
| `os.listdir()`                   | Explore dataset            |
| `image_dataset_from_directory()` | Load images                |
| `Rescaling()`                    | Normalize images           |
| `RandomFlip()`                   | Data augmentation          |
| `RandomRotation()`               | Data augmentation          |
| `RandomZoom()`                   | Data augmentation          |
| `Conv2D()`                       | Feature extraction         |
| `MaxPooling2D()`                 | Reduce feature dimensions  |
| `Flatten()`                      | Convert features to vector |
| `Dense()`                        | Classification             |
| `Dropout()`                      | Reduce overfitting         |
| `softmax`                        | Generate probabilities     |
| `model.compile()`                | Configure training         |
| `model.fit()`                    | Train model                |
| `model.evaluate()`               | Evaluate model             |
| `model.predict()`                | Generate predictions       |
| `np.argmax()`                    | Select predicted class     |
| `confusion_matrix()`             | Generate confusion matrix  |
| `classification_report()`        | Calculate metrics          |
| `kt.RandomSearch()`              | Hyperparameter tuning      |
| `model.save()`                   | Save trained model         |
---
