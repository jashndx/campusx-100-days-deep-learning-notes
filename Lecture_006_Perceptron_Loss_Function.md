# Lecture 006: Perceptron Loss Function | Hinge Loss | Sigmoid | BCE

> **CampusX 100 Days of Deep Learning** | Video ID: `2_gCL5RAkHc` | Duration: 35m 12s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=2_gCL5RAkHc) | **Transcript Status**: Available (en-US, 8051 words)

---

## 1. Executive Summary & Core Intuition
While the Perceptron Trick works via geometric heuristics, modern machine learning requires a differentiable 'Loss Function' that can be minimized systematically via Gradient Descent.

Why can't we use classification error count as a loss function? Because the number of misclassified points is a step-discontinuous integer function whose gradient is zero almost everywhere.

Rosenblatt proposed minimizing the sum of distances of misclassified points from the boundary.

Alternatively, replacing the step function with the smooth, differentiable **Sigmoid activation function** allows the model to output calibrated probabilities $\hat{y} \in (0, 1)$, giving rise to Logistic Regression and the **Binary Cross-Entropy (Log Loss)** loss function.


## 2. Key Definitions & Formal Terminology
- **Perceptron Criterion (Loss)**: The sum of the negative projections of misclassified samples onto the weight vector: $L(\mathbf{w}) = -\sum_{i \in \mathcal{M}} (\mathbf{w}^T \mathbf{x}_i + b) y_i$.
- **Sigmoid (Logistic) Function**: A smooth S-shaped mathematical activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ that squashes any real number into the open interval $(0, 1)$.
- **Binary Cross-Entropy (Log Loss)**: The negative log-likelihood loss function derived from Maximum Likelihood Estimation for Bernoulli-distributed binary classification targets.


## 3. Mathematical Formulations & Derivations
**1. Rosenblatt Perceptron Loss Formulation ($y_i \in \{-1, +1\}$):**

For misclassified samples $\mathcal{M}$, $y_i (\mathbf{w}^T \mathbf{x}_i + b) < 0$:

$$L(\mathbf{w}, b) = -\sum_{i \in \mathcal{M}} y_i (\mathbf{w}^T \mathbf{x}_i + b)$$



Gradient with respect to weights:

$$\nabla_{\mathbf{w}} L = -\sum_{i \in \mathcal{M}} y_i \mathbf{x}_i \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta \sum_{i \in \mathcal{M}} y_i \mathbf{x}_i$$



**2. Sigmoid Activation & Binary Cross Entropy ($y_i \in \{0, 1\}$):**

$$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}$$

$$\mathcal{L}_{BCE}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$



Derivative of Sigmoid:

$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$



Gradient of BCE Loss with respect to $w_j$:

$$\frac{\partial \mathcal{L}_{BCE}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_{ij}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Gradient Descent Optimization with BCE Loss:**

1. Initialize parameters $\mathbf{w} \sim \mathcal{N}(0, 0.01)$ and $b = 0$.

2. Compute linear combinations for batch: $\mathbf{z} = \mathbf{X}\mathbf{w} + b$.

3. Compute non-linear probability predictions: $\hat{\mathbf{y}} = \sigma(\mathbf{z})$.

4. Evaluate BCE Loss: $\mathcal{L} = -\frac{1}{N} \sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$.

5. Compute analytical gradient vector:

   $$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

   $$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

6. Update parameters simultaneously:

   $$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$


## 5. Implementation Code Snippet
```python

import numpy as np



def sigmoid(z):

    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))



def binary_cross_entropy(y_true, y_pred):

    eps = 1e-15

    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))



def train_logistic_perceptron(X, y, epochs=1000, lr=0.1):

    N, D = X.shape

    w = np.zeros(D)

    b = 0.0

    for epoch in range(epochs):

        z = np.dot(X, w) + b

        y_hat = sigmoid(z)

        loss = binary_cross_entropy(y, y_hat)

        

        # Gradients

        dw = (1/N) * np.dot(X.T, (y_hat - y))

        db = (1/N) * np.sum(y_hat - y)

        

        w -= lr * dw

        b -= lr * db

    return w, b

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is 0-1 loss (misclassification count) not used with Gradient Descent?
**Answer**:
0-1 loss is piecewise constant. Its derivative is zero wherever it is differentiable, and undefined at the transition thresholds. Gradient descent requires informative, non-zero gradient vectors ($\nabla L \neq 0$) that indicate the directional derivative of steepest descent.

### Q2: Derive the derivative of the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.
**Answer**:
$$\\frac{d}{dz} (1+e^{-z})^{-1} = -(1+e^{-z})^{-2} (-e^{-z}) = \\frac{e^{-z}}{(1+e^{-z})^2} = \\frac{1}{1+e^{-z}} \\cdot \\frac{e^{-z}}{1+e^{-z}} = \\sigma(z)(1 - \\sigma(z))$$

### Q3: Why is the gradient of Binary Cross-Entropy identical in form to the Mean Squared Error gradient of linear regression?
**Answer**:
Both belong to the Generalized Linear Model (GLM) family with canonical link functions. For the Bernoulli distribution, the canonical link is the logit function (inverse sigmoid); when paired with cross-entropy, the non-linearities in the derivative cancel out neatly to yield $(\hat{y}_i - y_i) x_{ij}$.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Know the derivation of $\\frac{d\\sigma(z)}{dz} = \\sigma(z)(1-\\sigma(z))$.
- BCE Gradient: $\\nabla_w L = \\frac{1}{N} X^T (\\hat{y} - y)$.
- Sigmoid maps $(-\\infty, +\\infty) \\to (0, 1)$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=2_gCL5RAkHc)
- [Lecture Video](https://www.youtube.com/watch?v=2_gCL5RAkHc)

---
