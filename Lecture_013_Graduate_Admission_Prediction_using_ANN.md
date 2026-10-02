# Lecture 013: Graduate Admission Prediction using ANN | Regression with ANN

> **CampusX 100 Days of Deep Learning** | Video ID: `RCmiPBiA4qg` | Duration: 27m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=RCmiPBiA4qg) | **Transcript Status**: Available (en-US, 2267 words)

---

## 1. Executive Summary & Core Intuition
While classification predicts discrete category probabilities, Regression predicts continuous real-valued targets (e.g., chance of graduate admission from $0.0$ to $1.0$, housing prices, stock returns).

This lecture demonstrates adapting an ANN for regression tasks:

1. Output Layer: Contains exactly 1 neuron with a **linear (identity) activation function** ($g(z) = z$) or Sigmoid if the output is strictly bounded in $[0, 1]$.

2. Loss Function: Shift from Cross-Entropy to **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**.

3. Evaluation Metrics: $R^2$ score, RMSE, MAE.


## 2. Key Definitions & Formal Terminology
- **Regression ANN**: A neural network architecture engineered to output continuous real-valued numerical variables.
- **Mean Squared Error (MSE)**: The average of the squared differences between predicted values and actual ground truth targets: $\\frac{1}{N}\\sum (y - \\hat{y})^2$.
- **Linear Activation Function**: An identity function $f(z) = z$ that passes the weighted sum directly to the output without compression or non-linear saturation.


## 3. Mathematical Formulations & Derivations
**Regression Loss Formulations:**



1. **Mean Squared Error (MSE / L2 Loss):**

   $$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

   Gradient with respect to prediction:

   $$\frac{\partial \mathcal{L}_{MSE}}{\partial \hat{y}_i} = -\frac{2}{N}(y_i - \hat{y}_i)$$



2. **Mean Absolute Error (MAE / L1 Loss):**

   $$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$



3. **Coefficient of Determination ($R^2$ Score):**

   $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Comparison: Classification vs Regression in ANNs**



| Component | Binary Classification | Multi-Class Classification | Regression |

| :--- | :--- | :--- | :--- |

| **Output Neurons** | 1 | $K$ (number of classes) | 1 (or $M$ for multi-output) |

| **Output Activation** | `sigmoid` | `softmax` | `linear` (None) or `relu` |

| **Loss Function** | `binary_crossentropy` | `categorical_crossentropy` | `mean_squared_error` / `mae` |

| **Primary Metric** | Accuracy, ROC-AUC, F1 | Accuracy, Top-k Accuracy | MSE, RMSE, MAE, $R^2$ |


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import r2_score



# Regression ANN Model

model = Sequential([

    Dense(16, activation='relu', input_dim=X_train.shape[1]),

    Dense(8, activation='relu'),

    Dense(1, activation='linear')  # Output layer for continuous regression

])



model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2)



y_pred = model.predict(X_test_scaled)

print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why must the output activation function be 'linear' for general regression?
**Answer**:
A linear activation $g(z) = z$ has an unbounded range $(-\\infty, +\\infty)$, allowing the network to predict any arbitrary real value. Using Sigmoid would bound outputs to $(0, 1)$ and ReLU would prevent negative predictions.

### Q2: Compare MSE and MAE loss functions in the presence of extreme outliers.
**Answer**:
MSE squares the error term $(y_i - \hat{y}_i)^2$, causing outliers to dominate the gradient and pull the model heavily towards anomalies. MAE penalizes errors linearly $|y_i - \hat{y}_i|$, making it much more robust to outliers.

### Q3: What does an $R^2$ score of 0.85 indicate?
**Answer**:
It means that 85% of the total variance in the dependent target variable is explained by the features and the neural network model, with the remaining 15% attributed to unexplained residual variance.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Output layer activation for regression is `linear` (or omitted).
- Loss is MSE or MAE, never cross-entropy.
- Evaluate using RMSE, MAE, or $R^2$ score.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=RCmiPBiA4qg)
- [Lecture Video](https://www.youtube.com/watch?v=RCmiPBiA4qg)
- [Kaggle GRE Admission Notebook](https://www.kaggle.com/campusx/gre-admission-prediction)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
