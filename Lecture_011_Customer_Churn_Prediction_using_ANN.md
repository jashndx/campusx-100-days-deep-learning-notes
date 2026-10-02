# Lecture 011: Customer Churn Prediction using ANN | Keras & TensorFlow

> **CampusX 100 Days of Deep Learning** | Video ID: `9wmImImmgcI` | Duration: 42m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=9wmImImmgcI) | **Transcript Status**: Available (hi, 5099 words)

---

## 1. Executive Summary & Core Intuition
This lecture delivers the end-to-end industry blueprint for deploying an Artificial Neural Network on structured tabular data.

Using the classic bank customer churn dataset, the workflow covers:

1. Exploratory Data Analysis & identifying target leakage.

2. Encoding categorical attributes (One-Hot Encoding for nominal variables, Label Encoding for binary).

3. Critical step: Feature scaling using StandardScaler (neural networks will fail or train extremely slowly if features have disparate numerical scales).

4. Train/Test splitting without data snooping.

5. Model architecture design, choosing Binary Cross-Entropy loss, Adam optimizer, and monitoring validation loss and accuracy curves to detect overfitting.


## 2. Key Definitions & Formal Terminology
- **Data Snooping / Leakage**: A fatal data science error where information from outside the training dataset (such as test set statistics during scaling) is inadvertently used to train the model.
- **One-Hot Encoding**: A transformation of categorical variables with $K$ categories into $K$ binary indicator columns.
- **Overfitting**: A modeling error where a neural network memorizes noise in the training set, characterized by diverging training loss (decreasing) and validation loss (increasing).


## 3. Mathematical Formulations & Derivations
**StandardScaler Standardization Formula:**

For feature column $j$:

$$\mu_j = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{N_{train}} \sum_{i=1}^{N_{train}} (x_{ij} - \mu_j)^2}$$

$$x_{ij}^{scaled} = \frac{x_{ij} - \mu_j}{\sigma_j}$$



Crucial exam rule: $\mu_j$ and $\sigma_j$ must be fitted *only* on the training split, and then used to transform both train and test splits:

$$\mathbf{X}_{test}^{scaled} = \frac{\mathbf{X}_{test} - \boldsymbol{\mu}_{train}}{\boldsymbol{\sigma}_{train}}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**ANN Architecture for Binary Churn Classification:**

- Input Layer: Shape $(D_{in},) = (11,)$ features after encoding.

- Hidden Layer 1: `Dense(11, activation='relu')`

- Hidden Layer 2: `Dense(11, activation='relu')`

- Output Layer: `Dense(1, activation='sigmoid')`

- Loss: `binary_crossentropy`

- Optimizer: `adam`

- Metrics: `['accuracy']`


## 5. Implementation Code Snippet
```python

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense



# 1. Split & Preprocess

# Assuming df loaded

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test) # Do NOT fit on test!



# 2. Build ANN

model = Sequential([

    Dense(11, activation='relu', input_dim=X_train_scaled.shape[1]),

    Dense(11, activation='relu'),

    Dense(1, activation='sigmoid')

])



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=32)



# 3. Evaluate

y_pred_prob = model.predict(X_test_scaled)

y_pred = (y_pred_prob > 0.5).astype(int)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why must `fit_transform` be called only on the training set and `transform` on the test set?
**Answer**:
Fitting on the test set causes data leakage. It pollutes the training phase with global distribution parameters ($\mu, \sigma$) of unseen evaluation data, providing overly optimistic and invalid evaluation metrics.

### Q2: Why is accuracy an inadequate evaluation metric for customer churn prediction?
**Answer**:
Customer churn datasets are typically highly class-imbalanced (e.g., 90% non-churn, 10% churn). A naive model predicting 'no churn' for all samples achieves 90% accuracy while having zero recall on the positive class. Metrics like Precision, Recall, F1-Score, and PR-AUC are required.

### Q3: What does the `history` object returned by `model.fit()` contain?
**Answer**:
It contains a dictionary (`history.history`) recording loss and evaluation metrics (e.g., `loss`, `val_loss`, `accuracy`, `val_accuracy`) evaluated at the conclusion of every single epoch, useful for diagnosing bias and variance.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Never fit scalers on test data.
- Choose classification threshold (default 0.5) based on precision-recall trade-offs.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=9wmImImmgcI)
- [Lecture Video](https://www.youtube.com/watch?v=9wmImImmgcI)
- [Kaggle Churn Notebook](https://www.kaggle.com/campusx/notebook8ad570467f)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
