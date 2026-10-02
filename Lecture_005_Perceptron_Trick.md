# Lecture 005: Perceptron Trick | How to train a Perceptron | Step by Step

> **CampusX 100 Days of Deep Learning** | Video ID: `Lu2bruOHN6g` | Duration: 26m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=Lu2bruOHN6g) | **Transcript Status**: Available (en-US, 8031 words)

---

## 1. Executive Summary & Core Intuition
How does a perceptron adjust its weights when it misclassifies a point?

This is demonstrated by the 'Perceptron Trick'. 

Consider a line $Ax + By + C = 0$. 

If a positive point $(p, q)$ with $y=1$ lies on the negative side (where $Ap + Bq + C < 0$), the line needs to shift towards $(p, q)$. 

To pull the line towards the point, we add the point's coordinates scaled by a learning rate $\eta$:

$A_{new} = A + \eta p, \ B_{new} = B + \eta q, \ C_{new} = C + \eta$.

If a negative point $(p, q)$ with $y=0$ lies on the positive side (where $Ap + Bq + C > 0$), we push the line away by subtracting:

$A_{new} = A - \eta p, \ B_{new} = B - \eta q, \ C_{new} = C - \eta$.

This simple algebraic operation rotates and shifts the hyperplane until all training points are correctly classified.


## 2. Key Definitions & Formal Terminology
- **Perceptron Learning Rule**: An iterative weight update algorithm where weights are modified only upon encountering misclassified training samples.
- **Learning Rate ($\eta$)**: A positive scalar hyperparameter ($0 < \eta \le 1$) that controls the step magnitude of the hyperplane adjustment during each update.
- **Perceptron Convergence Theorem**: Block and Novikoff's mathematical proof establishing that if the training data is linearly separable, the Perceptron Learning Algorithm is guaranteed to converge in a finite number of steps.


## 3. Mathematical Formulations & Derivations
**General Vectorized Perceptron Learning Rule:**

For a misclassified sample $(\mathbf{x}_i, y_i)$ where $y_i \in \{0, 1\}$ and $\hat{y}_i \in \{0, 1\}$:



$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$$

$$b^{(t+1)} = b^{(t)} + \eta (y_i - \hat{y}_i)$$



**Case Analysis:**

1. **Correctly Classified ($y_i = \hat{y}_i$):**

   $$y_i - \hat{y}_i = 0 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)}, \quad b^{(t+1)} = b^{(t)} \quad \text{(No update)}$$

2. **False Negative ($y_i = 1, \hat{y}_i = 0$):**

   $$y_i - \hat{y}_i = +1 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta \mathbf{x}_i, \quad b^{(t+1)} = b^{(t)} + \eta$$

3. **False Positive ($y_i = 0, \hat{y}_i = 1$):**

   $$y_i - \hat{y}_i = -1 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \mathbf{x}_i, \quad b^{(t+1)} = b^{(t)} - \eta$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Perceptron Training Algorithm (Epoch-by-Epoch):**

1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ or small random values, $b \leftarrow 0$.

2. For each epoch $e \in \{1, 2, \dots, \text{epochs}\}$:

   a. Set `misclassified = False`

   b. For each sample $(\mathbf{x}_i, y_i)$ in dataset:

      i. Calculate linear activation: $z_i = \mathbf{w}^T \mathbf{x}_i + b$.

      ii. Compute prediction: $\hat{y}_i = 1 \text{ if } z_i \ge 0 \text{ else } 0$.

      iii. If $\hat{y}_i \neq y_i$:

           $\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$

           $b \leftarrow b + \eta (y_i - \hat{y}_i)$

           `misclassified = True`

   c. If not `misclassified`: Stop early (converged).


## 5. Implementation Code Snippet
```python

import numpy as np



def perceptron_trick(X, y, epochs=1000, lr=0.01):

    # Add bias term column of 1s to input

    X = np.insert(X, 0, 1, axis=1)

    weights = np.ones(X.shape[1])

    

    for epoch in range(epochs):

        # Pick random point or iterate

        j = np.random.randint(0, X.shape[0])

        y_hat = 1 if np.dot(X[j], weights) >= 0 else 0

        if y[j] != y_hat:

            weights = weights + lr * (y[j] - y_hat) * X[j]

            

    return weights[0], weights[1:] # bias, weights

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: State the Perceptron Convergence Theorem and its prerequisite condition.
**Answer**:
The Perceptron Convergence Theorem states that if a dataset is linearly separable by a margin $\gamma > 0$, the perceptron algorithm will converge to a separating hyperplane in at most $k \le \left(\frac{R}{\gamma}\right)^2$ updates, where $R = \max \|\mathbf{x}_i\|$ is the radius of the data sphere.

### Q2: What happens if the perceptron trick is applied to non-linearly separable data?
**Answer**:
The algorithm will never converge. It enters an infinite loop, oscillating indefinitely between different suboptimal hyperplanes as it attempts to satisfy contradictory constraints.

### Q3: Why is the learning rate $\eta$ necessary in the update rule?
**Answer**:
Without $\eta$ (or if $\eta=1$), adding an entire data vector $\mathbf{x}_i$ can cause massive, erratic overshooting—violently flipping the orientation of the hyperplane and misclassifying previously correct points.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $\mathbf{w}_{new} = \mathbf{w}_{old} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$.
- Update occurs exclusively when there is a classification error.
- Convergence is guaranteed ONLY for linearly separable data.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=Lu2bruOHN6g)
- [Lecture Video](https://www.youtube.com/watch?v=Lu2bruOHN6g)

---
