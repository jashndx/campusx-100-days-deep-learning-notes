# Lecture 074: Scaled Dot Product Attention | Why Do We Scale by Sqrt(d_k)?

> **CampusX 100 Days of Deep Learning** | Video ID: `r7mAt0iVqwo` | Duration: 25m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=r7mAt0iVqwo) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Why did the authors of the Transformer divide the dot products by the square root of key dimension $\sqrt{d_k}$?

Why not use standard unscaled dot products $\mathbf{Q}\mathbf{K}^T$?

This lecture provides the mathematical proof:

When the dimension $d_k$ is large (e.g., $d_k = 64$ or $512$), computing the dot product between two independent random vectors involves summing $d_k$ individual products: $\sum_{i=1}^{d_k} q_i k_i$.

By the central limit theorem, the variance of this sum grows linearly with dimension: $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$.

For $d_k = 64$, standard deviation is $\sqrt{64} = 8$!

Values in the dot-product matrix blow up into large magnitudes ($+20, -25$).

Passing large values into the Softmax function pushes outputs into extreme saturated regions where the function is virtually flat!

As a result, the derivative of Softmax vanishes to zero ($\approx 0$), halting backpropagation gradient flow!

Dividing by $\sqrt{d_k}$ normalizes the variance strictly back to $1.0$, keeping gradients alive.


## 2. Key Definitions & Formal Terminology
- **Scaling Factor ($\frac{1}{\sqrt{d_k}}$)**: The normalization factor applied to dot products in self-attention to preserve unit variance regardless of vector dimension.
- **Softmax Saturation**: A pathology where large input logit magnitudes force one Softmax probability to $1.0$ and all others to $0.0$, driving derivatives to near-zero and freezing gradient flow.
- **Variance of Dot Product**: The mathematical property showing that the sum of $d$ independent product terms with zero mean and unit variance has a total variance of $d$.


## 3. Mathematical Formulations & Derivations
**Mathematical Proof that $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$:**



Assume components $q_i$ and $k_i$ are independent random variables with zero mean and unit variance:

$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$

$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$



Let scalar dot product be $Z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$.

1. **Expected Value:**

   $$\mathbb{E}[Z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$



2. **Variance of Product of Independent Variables:**

   $$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = (1)(1) = 1$$



3. **Variance of the Sum of $d_k$ Independent Variables:**

   $$\text{Var}(Z) = \text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = \mathbf{d_k}$$

   Standard deviation: $\sigma = \sqrt{d_k}$.



**Applying the Scaling Factor:**

If we scale $Z$ by $\frac{1}{\sqrt{d_k}}$:

$$\text{Var}\left( \frac{Z}{\sqrt{d_k}} \right) = \left(\frac{1}{\sqrt{d_k}}\right)^2 \text{Var}(Z) = \frac{1}{d_k} \cdot d_k = \mathbf{1.0}$$

The variance is normalized back to $1.0$, completely independent of dimension $d_k$!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Softmax Gradient Vanishing Dynamics:**

Recall Softmax derivative: $\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)$.

- If inputs are unscaled ($z_1 = 30, z_2 = -20$):

  $$\sigma_1 \approx 1.0, \quad \sigma_2 \approx 0.0$$

  $$\text{Derivative: } \sigma_1(1 - \sigma_1) \approx 1.0(1 - 1.0) = \mathbf{0.0} \implies \text{Gradients Vanish!}$$

- With scaling ($z_1 = 3.0, z_2 = -2.0$):

  $$\sigma_1 \approx 0.99, \sigma_2 \approx 0.01 \implies \text{Healthy gradient flow!}$$


## 5. Implementation Code Snippet
```python

import numpy as np



# Numerical simulation proving variance growth

d_k = 100

N_trials = 10000



q = np.random.randn(N_trials, d_k) # Mean 0, Var 1

k = np.random.randn(N_trials, d_k) # Mean 0, Var 1



dot_products = np.sum(q * k, axis=1)

print(f"Unscaled Variance (expected {d_k}): {np.var(dot_products):.2f}")



scaled_dot_products = dot_products / np.sqrt(d_k)

print(f"Scaled Variance (expected 1.0): {np.var(scaled_dot_products):.2f}")

# Output verifies: Unscaled Var ~ 100.0, Scaled Var ~ 1.0!

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Derive why the variance of the dot product of two $d$-dimensional random vectors is $d$.
**Answer**:
Let $Z = \sum_{i=1}^d q_i k_i$. For independent zero-mean unit-variance variables, $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \times 1 = 1$. Since the variance of the sum of independent random variables is the sum of their variances: $\text{Var}(Z) = \sum_{i=1}^d 1 = d$.

### Q2: Why does high variance in logits cause gradients to vanish in Softmax?
**Answer**:
When logits have large variance, the exponentiated values differ by orders of magnitude. The largest logit dominates the denominator, driving its Softmax probability $\sigma_k \to 1.0$ and all other probabilities $\sigma_j \to 0$. The derivative of Softmax is $\sigma_i(\delta_{ij} - \sigma_j)$, which evaluates to $1(1-1) = 0$ for the winner and $0(0) = 0$ for losers, driving all gradients to zero.

### Q3: Does Dot-Product Attention outperform Additive Attention for small $d_k$?
**Answer**:
For small dimensions $d_k$, additive attention and unscaled dot-product attention perform similarly. For large dimensions $d_k$, unscaled dot-product attention degrades significantly due to vanishing gradients, while scaled dot-product attention performs equally well and is significantly faster.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Be able to reproduce the proof: $\\text{Var}(\\sum q_i k_i) = d_k$.
- Dividing by $\\sqrt{d_k}$ normalizes variance to $1.0$.
- Prevents Softmax saturation and vanishing gradients during backprop.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=r7mAt0iVqwo)
- [Lecture Video](https://www.youtube.com/watch?v=r7mAt0iVqwo)

---
