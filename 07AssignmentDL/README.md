Transfer Learning for Plant Disease Classification
1. Problem Statement

Implement transfer learning using pre-trained deep learning models for image classification and compare their performance.

The models considered in this assignment are:

AlexNet
VGG16
ResNet50
EfficientNetB0

The models are used for classifying plant diseases using the PlantVillage dataset.

2. Objective

The main objectives of this assignment are:

Understand transfer learning for image classification.
Use CNN architectures for plant disease classification.
Load and preprocess image data.
Train and evaluate multiple deep learning models.
Analyze model performance using accuracy and loss.
Generate confusion matrices and classification reports.
Compare the performance of different CNN architectures.
Identify the best-performing model.
3. Assignment Agenda
Load PlantVillage Dataset
        ↓
Image Preprocessing
        ↓
Training / Validation Split
        ↓
Build CNN Models
        ↓
Apply Transfer Learning
        ↓
Train Models
        ↓
Evaluate Models
        ↓
Confusion Matrix
        ↓
Classification Report
        ↓
Compare Performance
        ↓
Conclusion

4. Dataset
PlantVillage Dataset

PlantVillage is an image dataset containing healthy and diseased plant leaf images.

The dataset used in this assignment contains:

20,638 images
15 classes
16,516 training images
4,122 validation images
80% training / 20% validation split

The dataset includes plant categories such as:

Pepper
Potato
Tomato

Some of the disease classes include:

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

The PlantVillage dataset was accessed directly in Google Colab using KaggleHub.

import kagglehub

path = kagglehub.dataset_download("emmarex/plantdisease")


The required dataset folder was selected using:

data_dir = os.path.join(path, "PlantVillage")


This allows the dataset to be accessed directly in the Colab runtime without manually uploading it to Google Drive.

6. Image Preprocessing

All images were resized to:

224 × 224 × 3

A batch size of 32 was used.

The dataset was divided into training and validation sets using an 80:20 split.

For the initial CNN data generator:

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

Important preprocessing operations
Resizing: Converts images to a common input size.
Rescaling: Converts pixel values from 0–255 to 0–1.
Validation split: Separates part of the dataset for validation.
Categorical labels: Used because the problem has multiple classes.
7. ImageDataGenerator

ImageDataGenerator is used to load images from folders and prepare batches for training.

Example:

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)


For validation:

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)


shuffle=False is used for validation so that predictions remain aligned with the actual class labels during evaluation.

8. Models Used
8.1 AlexNet

AlexNet is an early and influential convolutional neural network architecture.

It consists of:

Convolutional layers
Max-pooling layers
Fully connected layers
ReLU activation
Dropout
Softmax output

A simplified AlexNet architecture was implemented using Keras.

The final classification layer contains 15 neurons:

Dense(NUM_CLASSES, activation="softmax")

AlexNet in this Assignment

A standard ImageNet-pretrained AlexNet implementation is not provided by TensorFlow/Keras.

Therefore, AlexNet was implemented as a CNN and trained from scratch.

8.2 VGG16

VGG16 is a deep convolutional neural network architecture.

It mainly uses:

3×3 convolution filters
Max-pooling layers
Fully connected classification layers

VGG16 was loaded with ImageNet pre-trained weights:

VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The original ImageNet classifier was removed.

The pre-trained layers were frozen:

for layer in vgg_base.layers:
    layer.trainable = False


A new classifier was then added for the 15 PlantVillage classes.

8.3 ResNet50

ResNet50 is a 50-layer CNN architecture.

Its main feature is the use of residual connections, which help deeper networks learn effectively.

ResNet50 was loaded using ImageNet pre-trained weights:

ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The original classifier was removed and replaced with a classifier containing 15 output classes.

The pre-trained layers were frozen during training.

8.4 EfficientNetB0

EfficientNetB0 is a lightweight and efficient CNN architecture.

It provides a good balance between:

Accuracy
Model size
Computational cost

EfficientNetB0 was loaded with ImageNet pre-trained weights:

EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)


The original classifier was removed and replaced with a new classifier for the 15 PlantVillage classes.

9. Transfer Learning

Transfer learning uses knowledge learned by a model on a large dataset and applies it to a new problem.

For VGG16, ResNet50, and EfficientNetB0, ImageNet pre-trained weights were used.

The general process is:

Pre-trained ImageNet Model
          ↓
Remove Original Classifier
          ↓
Freeze Pre-trained Layers
          ↓
Add New Classifier
          ↓
Train on PlantVillage


The pre-trained CNN works as a feature extractor while the new classification layers learn to identify PlantVillage disease classes.

This reduces the amount of training required compared with training a large CNN completely from scratch.

10. Classifier Layers

For the pre-trained models, the original classifier was replaced with:

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

GlobalAveragePooling2D

Converts the feature maps into a compact feature vector.

Dense

Learns useful combinations of the extracted features.

Dropout

Helps reduce overfitting by randomly disabling neurons during training.

Softmax

Produces probabilities for all 15 classes.

11. Model Compilation

The models were compiled using the Adam optimizer and categorical cross-entropy loss.

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

Adam Optimizer

Adam is an optimization algorithm used to update the model weights during training.

Categorical Cross-Entropy

Used as the loss function for multi-class classification.

Accuracy

Measures the percentage of correctly classified images.

12. Training

The models were trained using:

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


The important training parameters were:

Parameter	Value
Image Size	224 × 224
Batch Size	32
Epochs	5
Learning Rate	0.0001
Number of Classes	15

During training, the model learns patterns from the training images and validation accuracy is used to monitor performance on unseen images.

13. Accuracy and Loss

Two important metrics were monitored during training.

Accuracy

Higher accuracy means more images are classified correctly.

Loss

Lower loss indicates that the predicted probabilities are closer to the correct classes.

Training and validation curves were plotted using Matplotlib.

Example:

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])


These graphs help visualize the learning behavior of each model.

14. Model Evaluation

After training, each model was evaluated using the validation dataset.

loss, accuracy = model.evaluate(
    validation_data,
    verbose=1
)


The validation accuracy was used as the main metric for comparing the models.

15. Predictions

Predictions were generated using:

predictions = model.predict(validation_data)


The predicted class was obtained using:

np.argmax(predictions, axis=1)


argmax() returns the class having the highest predicted probability.

16. Confusion Matrix

A confusion matrix compares the actual classes with the predicted classes.

Actual Class
     ↓
Predicted Class


The diagonal values represent correctly classified images.

The off-diagonal values represent incorrect classifications.

The confusion matrix helps identify which disease classes are being confused with one another.

It was generated using:

cm = confusion_matrix(
    true_labels,
    predictions
)


The matrix was visualized using a Seaborn heatmap.

17. Classification Report

The classification report provides detailed performance for every class.

It contains:

Precision
Recall
F1-score
Support
Precision

Shows how many predicted samples of a class were actually correct.

Recall

Shows how many actual samples of a class were correctly detected.

F1-score

Represents the balance between precision and recall.

Support

Represents the number of actual samples belonging to the class.

The report was generated using:

classification_report(
    true_labels,
    predictions,
    target_names=class_names
)

18. Performance Comparison

The final validation accuracies were:

Model	Validation Accuracy
ResNet50	94.30%
EfficientNetB0	93.26%
AlexNet	88.60%
VGG16	86.00%
Ranking
ResNet50 → 94.30%
EfficientNetB0 → 93.26%
AlexNet → 88.60%
VGG16 → 86.00%

ResNet50 achieved the highest validation accuracy in this experiment.

19. Technologies Used
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
20. Important ML/DL Functions Used
Function	Purpose
ImageDataGenerator()	Image preprocessing and splitting
flow_from_directory()	Loads images from class folders
Conv2D()	Performs convolution
MaxPooling2D()	Reduces feature map dimensions
Dense()	Fully connected layer
Dropout()	Helps reduce overfitting
GlobalAveragePooling2D()	Converts feature maps into vectors
Sequential()	Builds sequential CNN models
Model()	Builds functional models
Adam()	Optimizer
model.fit()	Trains models
model.evaluate()	Evaluates models
model.predict()	Generates predictions
confusion_matrix()	Creates confusion matrix
classification_report()	Generates evaluation metrics
np.argmax()	Finds predicted class
plt.plot()	Plots training curves
sns.heatmap()	Displays confusion matrix
21. Algorithm
Load the PlantVillage dataset.
Identify the 15 disease classes.
Resize images to 224 × 224.
Split images into training and validation sets.
Create the CNN models.
Load ImageNet weights for VGG16, ResNet50 and EfficientNetB0.
Freeze the pre-trained layers.
Remove the original classification layer.
Add a new 15-class classification layer.
Compile the models using Adam and categorical cross-entropy.
Train each model for 5 epochs.
Evaluate models on validation data.
Generate predictions.
Generate confusion matrices.
Generate classification reports.
Compare validation accuracy.
Identify the best-performing model.
22. Advantages
Provides automatic plant disease classification.
Transfer learning reduces training requirements.
Pre-trained models provide useful image features.
Multiple CNN architectures can be compared.
Confusion matrices provide class-level analysis.
Classification reports provide detailed performance metrics.
PlantVillage is suitable for multi-class plant disease classification.
23. Limitations
Only 5 epochs were used.
Results depend on the selected dataset version and preprocessing.
AlexNet was trained from scratch because a standard ImageNet-pretrained AlexNet model was not available through TensorFlow/Keras.
Some classes contain fewer images than others.
Performance on controlled PlantVillage images may differ from real-world field images.
Further fine-tuning may improve model performance.
24. Result

The final validation results were:

ResNet50 → 94.30%
EfficientNetB0 → 93.26%
AlexNet → 88.60%
VGG16 → 86.00%

ResNet50 achieved the best validation accuracy of 94.30%.

25. Conclusion

This assignment demonstrated image classification using CNN architectures and transfer learning on the PlantVillage dataset.

VGG16, ResNet50, and EfficientNetB0 used ImageNet pre-trained weights, while AlexNet was implemented and trained from scratch.

Among all tested models, ResNet50 achieved the highest validation accuracy of 94.30%, followed by EfficientNetB0 with 93.26%.

The experiment shows that pre-trained deep learning models can effectively extract useful image features and achieve strong performance on plant disease classification tasks.
