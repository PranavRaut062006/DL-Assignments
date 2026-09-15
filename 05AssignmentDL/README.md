# RNN vs LSTM vs GRU — IMDB Sentiment Classification
## 1. Problem Statement
The objective of this assignment is to perform **binary sentiment classification** on movie reviews.
The model needs to predict:
| Label | Sentiment |
|---|---|
| 0 | Negative |
| 1 | Positive |

Since movie reviews are sequential text data, **RNN-based architectures** are suitable for this problem.
The assignment implements and compares:
- SimpleRNN
- LSTM
- GRU
---
## 2. Dataset
### IMDB Movie Reviews Dataset
The IMDB dataset is a commonly used dataset for binary sentiment classification.
| Property | Value |
|---|---:|
| Total Reviews | 50,000 |
| Training Samples | 25,000 |
| Testing Samples | 25,000 |
| Classes | 2 |
| Negative Label | 0 |
| Positive Label | 1 |
| Vocabulary Used | 10,000 words |

The dataset is available directly through TensorFlow/Keras.
    from tensorflow.keras.datasets import imdb
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=10000
    )

---
## 3. Dataset Loading
    from tensorflow.keras.datasets import imdb
    vocab_size = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=vocab_size
    )
### Important Parameters
| Parameter | Meaning |
|---|---|
| `X_train` | Training reviews |
| `y_train` | Training labels |
| `X_test` | Testing reviews |
| `y_test` | Testing labels |
| `num_words=10000` | Uses the 10,000 most frequent words |

Result:

    Training samples: 25000
    Testing samples : 25000

---

## 4. Integer Encoding

The IMDB dataset already contains integer-encoded reviews.

Example:

    [1, 14, 22, 16, 43, 530, 973, ...]

Each integer represents a word ID.

Pipeline:

    Movie Review
         ↓
    Words
         ↓
    Integer IDs
         ↓
    Embedding
         ↓
    RNN / LSTM / GRU
         ↓
    Sentiment

The integer IDs themselves do not represent word importance. The Embedding layer learns meaningful vector representations.

---

## 5. Sequence Padding

Movie reviews have different lengths, so all sequences are converted to a fixed length.

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_length = 200

    X_train_padded = pad_sequences(
        X_train,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

    X_test_padded = pad_sequences(
        X_test,
        maxlen=max_length,
        padding='post',
        truncating='post'
    )

### Padding Configuration

| Setting | Purpose |
|---|---|
| `maxlen=200` | Every review has 200 positions |
| `padding='post'` | Adds zeros at the end |
| `truncating='post'` | Removes extra tokens from the end |

Example:

    Original:
    [12, 45, 78, 91]

    Padded:
    [12, 45, 78, 91, 0, 0]

Final shapes:

    Training data: (25000, 200)
    Testing data : (25000, 200)

---

## 6. Embedding Layer

Integer IDs are converted into dense numerical vectors.

Example:

    15  → [0.12, -0.43, 0.72, ...]
    100 → [0.31,  0.18, -0.26, ...]
    500 → [-0.51, 0.66, 0.14, ...]

The vectors are learned during training.

Configuration:

    embedding_dim = 64

    layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim
    )

### Embedding Parameters

    10,000 × 64 = 640,000

| Parameter | Value |
|---|---:|
| Vocabulary | 10,000 |
| Embedding Dimension | 64 |
| Embedding Parameters | 640,000 |

---

# 7. Common Hyperparameters

The same basic configuration was used for all three models.

| Hyperparameter | Value |
|---|---:|
| Vocabulary Size | 10,000 |
| Maximum Sequence Length | 200 |
| Embedding Dimension | 64 |
| Recurrent Units | 64 |
| Batch Size | 64 |
| Epochs | 5 |
| Optimizer | Adam |
| Loss | Binary Cross-Entropy |
| Output Activation | Sigmoid |

Keeping these values consistent makes the comparison between architectures more meaningful.

---

# 8. RNN

## What is RNN?

RNN stands for **Recurrent Neural Network**.

RNNs are designed for sequential data and maintain information from previous time steps through a hidden state.

    Word 1 → RNN
              ↓
    Word 2 → RNN
              ↓
    Word 3 → RNN
              ↓
    Word 4 → RNN
              ↓
          Prediction

### RNN Architecture

    rnn_model = keras.Sequential([
        layers.Input(shape=(max_length,)),

        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),

        layers.SimpleRNN(rnn_units),

        layers.Dense(1, activation='sigmoid')
    ])

Architecture:

    Input
      ↓
    Embedding
      ↓
    SimpleRNN
      ↓
    Dense + Sigmoid
      ↓
    Sentiment

---

# 9. LSTM

## What is LSTM?

LSTM stands for **Long Short-Term Memory**.

It is an improved RNN architecture designed to handle long-term dependencies more effectively.

Basic RNNs can suffer from the vanishing gradient problem.

LSTM uses a memory cell and gates to control information flow.

### Main LSTM Gates

| Gate | Purpose |
|---|---|
| Forget Gate | Discards unnecessary old information |
| Input Gate | Decides what new information to store |
| Output Gate | Controls information passed forward |

### LSTM Architecture

    lstm_model = keras.Sequential([
        layers.Input(shape=(max_length,)),

        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),

        layers.LSTM(rnn_units),

        layers.Dense(1, activation='sigmoid')
    ])

Architecture:

    Input
      ↓
    Embedding
      ↓
    LSTM
      ↓
    Dense + Sigmoid
      ↓
    Sentiment

LSTM has more parameters because of its multiple gates.

    RNN recurrent parameters  → 8,256
    LSTM recurrent parameters → 33,024

---

# 10. GRU

## What is GRU?

GRU stands for **Gated Recurrent Unit**.

GRU is another improved RNN architecture designed to handle long-term dependencies.

It is simpler than LSTM because it uses fewer gates and does not maintain a separate memory cell.

### GRU Gates

| Gate | Purpose |
|---|---|
| Update Gate | Controls how much previous information is retained |
| Reset Gate | Controls how much previous information is forgotten |

### GRU Architecture

    gru_model = keras.Sequential([
        layers.Input(shape=(max_length,)),

        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim
        ),

        layers.GRU(rnn_units),

        layers.Dense(1, activation='sigmoid')
    ])

Architecture:

    Input
      ↓
    Embedding
      ↓
    GRU
      ↓
    Dense + Sigmoid
      ↓
    Sentiment

---

# 11. RNN vs LSTM vs GRU

| Feature | RNN | LSTM | GRU |
|---|---|---|---|
| Architecture | Basic | Complex | Medium |
| Hidden State | Yes | Yes | Yes |
| Memory Cell | No | Yes | No |
| Gates | No | 3 main gates | 2 main gates |
| Long-Term Dependencies | Weak | Strong | Strong |
| Complexity | Low | High | Medium |
| Parameters | Lowest | Highest | Lower than LSTM |
| Performance in Experiment | Lowest | Medium | Highest |

---

# 12. Model Compilation

All models use:

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

### Adam Optimizer

Adam updates the model's trainable parameters to minimize the loss.

    Prediction
        ↓
    Loss Calculation
        ↓
    Adam Optimizer
        ↓
    Weight Update
        ↓
    Better Prediction

### Binary Cross-Entropy

Because this is a binary classification problem:

    0 → Negative
    1 → Positive

`binary_crossentropy` is used as the loss function.

---

# 13. Model Training

### RNN

    rnn_history = rnn_model.fit(
        X_train_padded,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

### LSTM

    lstm_history = lstm_model.fit(
        X_train_padded,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )

### GRU

    gru_history = gru_model.fit(
        X_train_padded,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size
    )

### Training Configuration

| Setting | Value |
|---|---:|
| Training Data | 80% |
| Validation Data | 20% |
| Epochs | 5 |
| Batch Size | 64 |

---

# 14. Training History

Training history stores information from every epoch.

It includes:

- Training accuracy
- Training loss
- Validation accuracy
- Validation loss

Example:

    rnn_history.history['accuracy']
    rnn_history.history['val_accuracy']

These values are used to visualize:

- Learning progress
- Generalization
- Overfitting
- Underfitting
- Training instability

The RNN showed a large training-validation gap, indicating poor generalization.

---

# 15. Training Time

Training time was measured using:

    start_time = time.time()

    # Model training

    training_time = time.time() - start_time

Observed training times:

| Model | Training Time |
|---|---:|
| RNN | 37.25 s |
| LSTM | 24.76 s |
| GRU | 22.85 s |

In this particular Colab execution:

    GRU < LSTM < RNN

Training time depends on hardware, TensorFlow version, batch size, epochs, and system load, so these values are observations from this experiment.

---

# 16. Model Evaluation

The trained models were evaluated on unseen test data.

    model.evaluate(
        X_test_padded,
        y_test,
        batch_size=batch_size
    )

Test performance provides an indication of how well the model generalizes.

---

# 17. Predictions

Predictions were generated using:

    probabilities = model.predict(
        X_test_padded,
        batch_size=batch_size
    )

The sigmoid output gives probabilities between 0 and 1.

Conversion to classes:

    predictions = (
        probabilities >= 0.5
    ).astype(int).flatten()

Classification rule:

| Probability | Class |
|---|---|
| `< 0.5` | 0 → Negative |
| `>= 0.5` | 1 → Positive |

---

# 18. Evaluation Metrics

The following Scikit-learn functions were used:

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        classification_report
    )

| Metric | Meaning | Formula |
|---|---|---|
| Accuracy | Overall correct predictions | Correct / Total |
| Precision | Correct positive predictions | TP / (TP + FP) |
| Recall | Actual positives correctly identified | TP / (TP + FN) |
| F1 | Balance between precision and recall | 2PR / (P + R) |

Functions:

    accuracy_score(y_test, predictions)

    precision_score(y_test, predictions)

    recall_score(y_test, predictions)

    f1_score(y_test, predictions)

    confusion_matrix(y_test, predictions)

    classification_report(
        y_test,
        predictions,
        target_names=['Negative', 'Positive']
    )

---

# 19. Confusion Matrix

A confusion matrix contains:

                    Predicted
                 Negative Positive

Actual Negative     TN       FP

Actual Positive     FN       TP

A strong classifier should have large values along the main diagonal:

    TN       FP
    FN       TP

The confusion matrix was visualized using:

    sns.heatmap(...)

GRU produced a much stronger diagonal pattern than RNN and LSTM.

---

# 20. Model Parameters

Final parameter counts:

| Model | Total Parameters |
|---|---:|
| RNN | 648,321 |
| LSTM | 673,089 |
| GRU | 665,025 |

The Embedding layer alone contributes:

    640,000 parameters

The remaining parameters belong to the recurrent and Dense layers.

---

# 21. Final Experimental Results

| Model | Accuracy | Precision | Recall | F1-Score | Test Loss | Training Time |
|---|---:|---:|---:|---:|---:|---:|
| RNN | 50.95% | 50.92% | 52.92% | 51.90% | 0.8366 | 37.25 s |
| LSTM | 56.81% | 54.50% | 82.41% | 65.61% | 0.6674 | 24.76 s |
| GRU | **84.22%** | **81.13%** | **89.20%** | **84.97%** | **0.3807** | **22.85 s** |

---

# 22. Class-wise Performance

| Model | Negative F1 | Positive F1 | Observation |
|---|---:|---:|---|
| RNN | 0.50 | 0.52 | Close to random classification |
| LSTM | 0.42 | 0.66 | Better at positive reviews |
| GRU | **0.83** | **0.85** | Strong and balanced performance |

---

# 23. RNN Results

| Metric | Result |
|---|---:|
| Accuracy | 50.95% |
| Precision | 50.92% |
| Recall | 52.92% |
| F1-score | 51.90% |
| Test Loss | 0.8366 |

The RNN training accuracy increased, but validation accuracy remained close to 50%.

This indicates poor generalization and overfitting.

---

# 24. LSTM Results

| Metric | Result |
|---|---:|
| Accuracy | 56.81% |
| Precision | 54.50% |
| Recall | 82.41% |
| F1-score | 65.61% |
| Test Loss | 0.6674 |

LSTM performed better than the basic RNN.

Its positive-class recall was particularly high:

    Positive Recall = 82.41%

However, performance on negative reviews was weaker.

---

# 25. GRU Results

| Metric | Result |
|---|---:|
| Accuracy | **84.22%** |
| Precision | **81.13%** |
| Recall | **89.20%** |
| F1-score | **84.97%** |
| Test Loss | **0.3807** |
| Training Time | **22.85 s** |

GRU achieved the strongest overall results in this experiment.

---

# 26. Why GRU Performed Best

Possible reasons:

- Gating mechanisms preserve useful information.
- Better handling of long-term dependencies than basic RNN.
- Simpler architecture than LSTM.
- Fewer recurrent parameters than LSTM.
- Selected hyperparameters may have suited GRU.
- The 5-epoch configuration produced stronger GRU performance.

However, GRU is not guaranteed to outperform LSTM on every dataset.

Performance depends on:

- Dataset
- Hyperparameters
- Sequence length
- Number of units
- Training duration
- Optimization settings
- Random initialization

Therefore:

> **GRU performed best in this particular experiment.**

---

# 27. Complete Pipeline

    IMDB Dataset
         ↓
    Train/Test Split
         ↓
    Integer Encoded Reviews
         ↓
    Sequence Padding
         ↓
    Embedding Layer
         ↓
     ┌──────────┼──────────┐
     ↓          ↓          ↓
    RNN        LSTM       GRU
     ↓          ↓          ↓
    Dense      Dense      Dense
     ↓          ↓          ↓
    Sigmoid    Sigmoid    Sigmoid
     ↓          ↓          ↓
    Prediction Prediction Prediction
     └──────────┼──────────┘
                ↓
          Model Evaluation
                ↓
    Accuracy / Precision / Recall / F1
                ↓
          Confusion Matrix
                ↓
          Model Comparison

---

# 28. Important TensorFlow/Keras Functions

| Function | Purpose |
|---|---|
| `imdb.load_data()` | Loads IMDB dataset |
| `pad_sequences()` | Makes sequences equal length |
| `keras.Sequential()` | Creates sequential model |
| `layers.Input()` | Defines input shape |
| `layers.Embedding()` | Converts IDs to vectors |
| `layers.SimpleRNN()` | Creates basic RNN |
| `layers.LSTM()` | Creates LSTM |
| `layers.GRU()` | Creates GRU |
| `layers.Dense()` | Fully connected layer |
| `model.compile()` | Configures model |
| `model.fit()` | Trains model |
| `model.evaluate()` | Evaluates model |
| `model.predict()` | Generates predictions |
| `model.summary()` | Displays architecture |
| `model.count_params()` | Counts parameters |

---

# 29. Important Python Libraries

| Library | Usage |
|---|---|
| NumPy | Numerical operations |
| Pandas | Comparison DataFrame |
| Matplotlib | Training and comparison graphs |
| Seaborn | Confusion matrix visualization |
| `time` | Training time measurement |
| Scikit-learn | Evaluation metrics |
| TensorFlow/Keras | Deep learning models |

---

# 30. Visualizations

The assignment includes visualizations for:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Confusion matrices
- Model performance comparison
- Training time comparison

These visualizations help analyze:

    Learning
       ↓
    Generalization
       ↓
    Overfitting
       ↓
    Model Performance
       ↓
    Model Comparison

---

# 31. Advantages and Limitations

| Model | Advantages | Limitations |
|---|---|---|
| RNN | Simple, fewer parameters, easy to understand | Vanishing gradient, weak long-term dependency handling |
| LSTM | Strong long-term dependency handling, gates, memory cell | More parameters, more complex |
| GRU | Gated architecture, simpler than LSTM, fewer recurrent parameters | Performance depends on hyperparameters |

---

# 32. Deep Learning Concepts Demonstrated

- Sequence Classification
- Integer Encoding
- Sequence Padding
- Word Embeddings
- Recurrent Neural Networks
- SimpleRNN
- LSTM
- GRU
- Hidden State
- Gating Mechanisms
- Sigmoid Activation
- Binary Classification
- Binary Cross-Entropy
- Adam Optimization
- Backpropagation Through Time
- Epochs
- Batch Size
- Validation Split
- Overfitting
- Generalization
- Model Parameters
- Training Time
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Classification Report

---

# 33. Technologies Used

- Python
- Google Colab
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# 34. Final Result

Final test accuracy:

| Model | Accuracy |
|---|---:|
| RNN | 50.95% |
| LSTM | 56.81% |
| **GRU** | **84.22%** |

GRU achieved:

| Metric | GRU |
|---|---:|
| Accuracy | **84.22%** |
| Precision | **81.13%** |
| Recall | **89.20%** |
| F1-score | **84.97%** |

### Best Model

    Best Model = GRU

for this particular IMDB sentiment classification experiment.

---

# 35. Conclusion

In this assignment, **RNN, LSTM, and GRU** were implemented and compared for binary sentiment classification using the **IMDB Movie Reviews dataset**.

The reviews were integer encoded and padded to a fixed length of **200 tokens**. An **Embedding layer** converted word IDs into dense vector representations.

All three models used the same preprocessing, embedding dimension, recurrent units, batch size, epochs, optimizer, and loss function to make the comparison meaningful.

The results were:

    RNN  → 50.95%
    LSTM → 56.81%
    GRU  → 84.22%

GRU achieved the best performance with:

    Accuracy  → 84.22%
    Precision → 81.13%
    Recall    → 89.20%
    F1-score  → 84.97%

The experiment demonstrates the importance of gated recurrent architectures for sequence learning.

**GRU provided the strongest performance in this particular experiment while also having fewer parameters than LSTM and the lowest observed training time.**

Overall, this assignment demonstrates how **RNN, LSTM, and GRU** can be used for sequence classification and how different evaluation metrics can be used to compare deep learning models.