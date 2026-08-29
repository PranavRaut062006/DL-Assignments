Absolutely — here is the complete content formatted as a clean README.md file.

Writing
Transfer Learning for Plant Disease Classification
1. Problem Statement

Implement image classification using deep learning models and compare their performance.

The assignment focuses on using CNN-based models for classifying plant leaf images into different disease categories. The models used are:

AlexNet
VGG16
ResNet50
EfficientNetB0

The models are trained and evaluated using the PlantVillage dataset.

2. Objective

The main objectives of this assignment are:

To understand transfer learning for image classification.
To use pre-trained CNN models for plant disease classification.
To prepare and preprocess image data.
To train multiple deep learning models.
To evaluate models using accuracy, loss, confusion matrix, and classification report.
To compare the performance of different CNN architectures.
To identify the best-performing model.
3. Project Overview

This project performs multi-class plant disease classification using the PlantVillage dataset.

The general workflow is:

PlantVillage Dataset
        ↓
Image Preprocessing
        ↓
Training / Validation Split
        ↓
Model Selection
        ↓
Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Confusion Matrix
        ↓
Classification Report
        ↓
Performance Comparison


Four CNN architectures are evaluated under the same general experimental setup.

VGG16, ResNet50, and EfficientNetB0 use ImageNet pre-trained weights. AlexNet is implemented as a CNN architecture and trained from scratch because a standard ImageNet-pretrained AlexNet model is not provided by TensorFlow/Keras.

4. Dataset
PlantVillage Dataset

PlantVillage is an image dataset containing photographs of plant leaves belonging to different healthy and diseased categories.

In this experiment, the downloaded PlantVillage version contains:

Dataset Information	Value
Total Images	20,638
Number of Classes	15
Training Images	16,516
Validation Images	4,122
Training / Validation Split	80% / 20%

The classes include diseases and healthy categories from plants such as:

Pepper
Potato
Tomato
Classes

Examples of classes include:

Pepper__bell___Bacterial_spot
Pepper__bell___healthy
Potato___Early_blight
Potato___Late_blight
Potato___healthy
Tomato_Bacterial_spot
Tomato_Early_blight
Tomato_Late_blight
Tomato_Leaf_Mold
Tomato_Septoria_leaf_spot
Tomato__Target_Spot
Tomato__Tomato_YellowLeaf__Curl_Virus
Tomato__Tomato_mosaic_virus
Tomato_Spider_mites_Two_spotted_spider_mite
Tomato_healthy

5. Dataset Loading

The dataset was accessed directly in the Google Colab environment using KaggleHub.

import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")


The dataset was then located using:

data_dir = os.path.join(path, "PlantVillage")


This avoids manually downloading and uploading the dataset to Google Drive.

6. Image Preprocessing

Images were resized to:

224 × 224 × 3


The same image size was used for the CNN models to maintain a consistent input format.

The dataset was divided using:

validation_split=0.2


This creates:

80% → Training
20% → Validation

7. ImageDataGenerator

ImageDataGenerator was used to load images from folders and create batches.

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

Important Parameters
Parameter	Purpose
rescale	Normalizes pixel values
validation_split	Creates the validation portion
target_size	Resizes images
batch_size	Controls images processed at once
class_mode="categorical"	Used for multi-class classification
shuffle=True	Used for training data
shuffle=False	Used for validation data
8. Training Parameters

Common parameters used in the experiment:

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.0001
SEED = 42


These settings were kept consistent where applicable to make model comparison easier.

9. Models Used
9.1 AlexNet

AlexNet is an early deep convolutional neural network architecture.

It contains:

Convolutional layers
Max-pooling layers
Fully connected layers
ReLU activation
Dropout
Softmax output layer

The final layer was changed to:

Dense(NUM_CLASSES, activation="softmax")


Since there are 15 classes, the output layer contains 15 neurons.

AlexNet in this Assignment

A standard ImageNet-pretrained AlexNet implementation is not available in TensorFlow/Keras applications. Therefore, AlexNet was implemented using Keras layers and trained from scratch.

9.2 VGG16

VGG16 is a deep CNN architecture containing 16 layers.

It mainly uses:

3×3 convolution filters
Max pooling
Fully connected classification layers

For this assignment, VGG16 was loaded with ImageNet weights:

VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The original classifier was removed and replaced with a classifier for 15 PlantVillage classes.

The pre-trained layers were frozen:

for layer in vgg_base.layers:
    layer.trainable = False

9.3 ResNet50

ResNet50 is a 50-layer CNN that uses residual connections.

Residual connections help deeper networks learn effectively by allowing information to pass through shortcut connections.

ResNet50 was loaded using:

ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The ImageNet classification layer was removed and a new classifier was added for the 15 PlantVillage classes.

The pre-trained layers were frozen during training.

9.4 EfficientNetB0

EfficientNetB0 is a lightweight CNN architecture designed to provide a good balance between accuracy and computational efficiency.

It was loaded with ImageNet weights:

EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The original classifier was replaced with a new classifier containing 15 output classes.

The pre-trained layers were frozen during the initial training.

10. Transfer Learning

Transfer learning uses knowledge learned from a large dataset and applies it to another classification problem.

In this project:

ImageNet Pre-trained Model
          ↓
Remove Original Classifier
          ↓
Freeze Pre-trained Layers
          ↓
Add New Classifier
          ↓
Train on PlantVillage


The pre-trained CNN acts as a feature extractor.

The newly added layers learn to classify the PlantVillage disease classes.

This reduces training requirements compared with training a large CNN completely from scratch.

11. Model Classifier

A common classifier structure was used for the pre-trained models:

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

GlobalAveragePooling2D

Converts the feature maps produced by the CNN into a compact feature vector.

Dense Layer

Learns combinations of extracted features.

Dropout

Reduces overfitting by randomly disabling some neurons during training.

Softmax

Produces probability values for all 15 classes.

12. Model Compilation

The models were compiled using:

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

Adam Optimizer

Adam adjusts model weights during training using adaptive learning rates.

Categorical Cross-Entropy

Used as the loss function because this is a multi-class classification problem.

Accuracy

Measures the percentage of correctly classified images.

13. Model Training

The models were trained using:

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


During training, the model learns patterns from the training images.

Validation data is used to measure how well the model performs on unseen images.

14. Accuracy and Loss

Two important metrics were monitored:

Accuracy

Higher accuracy indicates more correct predictions.

Loss

Lower loss indicates that the predicted probabilities are closer to the correct class.

Training and validation accuracy/loss were plotted using Matplotlib.

Example:

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])


These graphs help understand the learning behavior of the model.

15. Model Evaluation

After training, each model was evaluated on the validation dataset.

loss, accuracy = model.evaluate(
    validation_data,
    verbose=1
)


The final validation accuracy was used for comparing the models.

16. Predictions

Predictions were generated using:

predictions = model.predict(validation_data)


The predicted class was obtained using:

np.argmax(predictions, axis=1)


argmax() returns the class having the highest predicted probability.

17. Confusion Matrix

A confusion matrix is used to understand the classification performance for each class.

It compares:

Actual Class
     vs
Predicted Class


The diagonal values represent correctly classified images.

Off-diagonal values represent incorrect classifications.

A confusion matrix helps identify which plant diseases are being confused with each other.

It was generated using:

cm = confusion_matrix(
    true_labels,
    predictions
)


The matrix was visualized using a Seaborn heatmap.

Example:

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

18. Classification Report

The classification report provides detailed performance for every class.

It contains:

Precision
Recall
F1-score
Support
Precision

Measures how many predicted samples of a class were actually correct.

Recall

Measures how many actual samples of a class were correctly detected.

F1-score

The harmonic mean of precision and recall.

Support

Number of actual samples belonging to a class.

The report was generated using:

classification_report(
    true_labels,
    predictions,
    target_names=class_names
)

19. Performance Comparison

The final validation accuracy obtained from the models was:

Rank	Model	Validation Accuracy
1	ResNet50	94.30%
2	EfficientNetB0	93.26%
3	AlexNet	88.60%
4	VGG16	86.00%
Performance Visualization
ResNet50       ███████████████████ 94.30%
EfficientNetB0 ██████████████████  93.26%
AlexNet        █████████████████   88.60%
VGG16          ████████████████    86.00%


ResNet50 achieved the highest validation accuracy in this experiment.

EfficientNetB0 performed very closely behind ResNet50 while being a more lightweight architecture.

20. Advantages
Uses deep learning for automatic disease classification.
Transfer learning reduces training requirements.
Pre-trained models provide useful image features.
Multiple CNN architectures can be compared.
Confusion matrices provide class-level analysis.
Classification reports provide precision, recall, and F1-score.
PlantVillage provides a suitable dataset for plant disease classification.
21. Limitations
Only 5 epochs were used for training.
Results depend on the selected dataset version and preprocessing.
AlexNet was trained from scratch rather than using pre-trained ImageNet weights.
Some disease classes contain fewer images than others.
Validation accuracy may not represent performance on real-world field images.
More training and fine-tuning could potentially improve performance.
22. Technologies Used
Python
Google Colab
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
KaggleHub
23. Libraries and Important Functions
Function / Class	Purpose
ImageDataGenerator()	Image preprocessing and data splitting
flow_from_directory()	Loads images from class folders
Conv2D()	Performs convolution
MaxPooling2D()	Reduces feature map size
Dense()	Fully connected layer
Dropout()	Reduces overfitting
GlobalAveragePooling2D()	Converts feature maps into feature vectors
Model()	Creates the final neural network
Sequential()	Creates sequential CNN models
Adam()	Optimization algorithm
model.fit()	Trains the model
model.evaluate()	Evaluates the model
model.predict()	Generates predictions
confusion_matrix()	Creates confusion matrix
classification_report()	Generates classification metrics
np.argmax()	Finds the predicted class
plt.plot()	Plots training curves
sns.heatmap()	Displays confusion matrix
24. Algorithm
Load the PlantVillage dataset.
Identify the plant disease classes.
Resize images to 224 × 224.
Split the dataset into training and validation sets.
Create the CNN models.
Load ImageNet weights for VGG16, ResNet50, and EfficientNetB0.
Freeze the pre-trained layers.
Replace the original classifier with a 15-class classifier.
Compile the models using Adam and categorical cross-entropy.
Train each model for 5 epochs.
Evaluate each model on validation data.
Generate predictions.
Generate confusion matrices.
Generate classification reports.
Compare validation accuracy.
Identify the best-performing model.
25. Result

The models achieved the following validation accuracy:

Model	Validation Accuracy
ResNet50	94.30%
EfficientNetB0	93.26%
AlexNet	88.60%
VGG16	86.00%
Best Performing Model

ResNet50 achieved the best performance with 94.30% validation accuracy.

The performance ranking was:

1. ResNet50       → 94.30%
2. EfficientNetB0 → 93.26%
3. AlexNet        → 88.60%
4. VGG16          → 86.00%

26. Conclusion

This assignment demonstrated the use of CNN architectures for multi-class plant disease classification using the PlantVillage dataset.

Transfer learning with pre-trained CNN models provided strong classification performance. Among the evaluated models, ResNet50 performed best, achieving 94.30% validation accuracy, followed by EfficientNetB0 with 93.26%.

The experiment shows that pre-trained deep learning models can effectively extract useful visual features for plant disease classification and can be adapted to new image classification tasks with relatively limited training.

Overall, ResNet50 was the best-performing model in this experiment, while EfficientNetB0 provided a strong alternative with competitive accuracy and a lightweight architecture.