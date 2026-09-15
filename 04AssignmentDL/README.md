# Assignment 4 — LSTM-Based Time-Series Forecasting

## 📌 Problem Statement

Develop an LSTM-based deep learning model for time-series forecasting using historical stock-price data. The model learns patterns from previous stock prices and predicts the next closing price.

---

## 🎯 Objective

The objective of this assignment is to understand how **Long Short-Term Memory (LSTM)** networks can be used for time-series forecasting. The assignment covers data collection, preprocessing, sequence generation, LSTM model development, training, prediction, and model evaluation.

---

## 📊 Dataset

For this assignment, historical **Apple Inc. (AAPL)** stock-price data is used.

The dataset is obtained directly from a public GitHub repository using Python, so no manual dataset download or Google Drive upload is required.

### Dataset Features

| Feature | Description |
|---|---|
| Date | Trading date |
| AAPL.Open | Opening stock price |
| AAPL.High | Highest price of the day |
| AAPL.Low | Lowest price of the day |
| AAPL.Close | Closing stock price |
| AAPL.Volume | Number of shares traded |
| AAPL.Adjusted | Adjusted closing price |
| dn | Lower price band |
| mavg | Moving average |
| up | Upper price band |
| direction | Price movement direction |

For forecasting, only **AAPL.Close** is used as the main feature.

### Dataset Size

- Total observations: **506**
- Training observations: **404**
- Testing observations: **102**

---

## 🧠 Why LSTM?

**LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) designed for sequential and time-series data.

Stock prices are sequential because the order of observations matters. LSTM can learn relationships between previous observations and use these learned patterns to generate future predictions.

LSTM is useful because it can retain important information from previous time steps through its internal memory mechanism.

---

## 🔄 Project Workflow

```text
Historical Stock Data
        ↓
Data Exploration
        ↓
Feature Selection
        ↓
Min-Max Scaling
        ↓
Chronological Train/Test Split
        ↓
60-Day Sequence Creation
        ↓
Reshape Data for LSTM
        ↓
Build LSTM Model
        ↓
Compile Model
        ↓
Train Model
        ↓
Generate Predictions
        ↓
Inverse Scaling
        ↓
Model Evaluation
        ↓
Visualization