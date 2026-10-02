# Lecture 030: Xavier/Glorot and He Weight Initialization in Deep Learning

> **CampusX 100 Days of Deep Learning** | Video ID: `nwVOSgcrbQI` | Duration: 33m 50s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=nwVOSgcrbQI) | **Transcript Status**: Available (en-US, 2891 words)

---

## 1. Executive Summary & Core Intuition
This lecture provides the full mathematical derivations of the two seminal initialization strategies that made modern deep learning work:

1. **Xavier (Glorot) Initialization (2010):** Engineered specifically for symmetric, zero-centered activations (Sigmoid and Tanh). It derives the weight variance required to keep both the forward activation variance and backward gradient variance constant, arriving at $\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$.

2. **He (Kaiming) Initialization (2015):** Xavier assumes linear/symmetric activations. ReLU zeroes out half of the inputs ($z < 0$), halving the variance at each layer! Kaiming He derived that for ReLU, the variance must be doubled: $\text{Var}(w) = \frac{2}{n_{in}}$.

Using He initialization with ReLU is the universal default in modern computer vision and deep learning.


## 2. Key Definitions & Formal Terminology
- **Xavier / Glorot Initialization**: Weight initialization designed for Sigmoid/Tanh that draws weights from a distribution with variance $\\text{Var}(W) = \\frac{2}{n_{in} + n_{out}}$.
- **He / Kaiming Initialization**: Weight initialization designed for ReLU/LeakyReLU that draws weights from a distribution with variance $\\text{Var}(W) = \\frac{2}{n_{in}}$.
- **LeCun Initialization**: Weight initialization designed for SELU activations where $\\text{Var}(W) = \\frac{1}{n_{in}}$.


## 3. Mathematical Formulations & Derivations
**Mathematical Derivation of Xavier Initialization (Glorot & Bengio, 2010):**



Forward pass variance: $\text{Var}(z^{[l]}) = n_{in} \text{Var}(w^{[l]}) \text{Var}(a^{[l-1]})$.

To ensure $\text{Var}(z^{[l]}) = \text{Var}(a^{[l-1]})$, forward pass requires:

$$\text{Var}(w) = \frac{1}{n_{in}}$$



Backward pass gradient variance: $\text{Var}(\delta^{[l-1]}) = n_{out} \text{Var}(w^{[l]}) \text{Var}(\delta^{[l]})$.

To ensure backward gradient variance is preserved:

$$\text{Var}(w) = \frac{1}{n_{out}}$$



Harmonic mean compromise:

$$\text{Var}(w) = \frac{2}{n_{in} + n_{out}}$$



- **Xavier Normal:** $\mathbf{W} \sim \mathcal{N}\left(0, \ \sigma = \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$

- **Xavier Uniform:** $\mathbf{W} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \ \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$



**He Initialization Derivation (He et al., 2015):**

For ReLU, since negative inputs are zeroed: $\mathbb{E}[a^2] = \frac{1}{2} \text{Var}(z)$.

Therefore, $\text{Var}(z^{[l]}) = n_{in} \text{Var}(w^{[l]}) \cdot \frac{1}{2} \text{Var}(z^{[l-1]})$.

To maintain $\text{Var}(z^{[l]}) = \text{Var}(z^{[l-1]})$:

$$\text{Var}(w) = \frac{2}{n_{in}}$$



- **He Normal:** $\mathbf{W} \sim \mathcal{N}\left(0, \ \sigma = \sqrt{\frac{2}{n_{in}}}\right)$

- **He Uniform:** $\mathbf{W} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \ \sqrt{\frac{6}{n_{in}}}\right)$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Activation & Initialization Pairing Rule Table:**



| Activation Function | Recommended Initializer | Distribution Formula |

| :--- | :--- | :--- |

| **Sigmoid / Logistic** | Xavier / Glorot Normal | $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{in} + n_{out}}}\right)$ |

| **Tanh** | Xavier / Glorot Uniform | $\mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$ |

| **ReLU / Leaky ReLU** | He / Kaiming Normal | $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$ |

| **SELU** | LeCun Normal | $\mathcal{N}\left(0, \sqrt{\frac{1}{n_{in}}}\right)$ |


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Best-practice pairing in Keras

model = models.Sequential([

    # ReLU paired with He Normal

    layers.Dense(128, activation='relu', kernel_initializer='he_normal', input_shape=(50,)),

    layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),

    

    # Tanh paired with Glorot Normal

    layers.Dense(32, activation='tanh', kernel_initializer='glorot_normal'),

    

    # Output layer

    layers.Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why did Xavier initialization fail when applied to deep ReLU networks?
**Answer**:
Xavier assumes the activation function is linear around zero with derivative 1 (like Tanh/Sigmoid near 0). ReLU zeroes out all negative activations, cutting the signal variance in half at each layer. Across a 20-layer network, Xavier causes the variance of activations to diminish by $0.5^{20} \approx 10^{-6}$, reintroducing vanishing gradients.

### Q2: What is the uniform distribution bound for He initialization?
**Answer**:
For a continuous uniform distribution $\mathcal{U}(-a, a)$, variance is $\text{Var} = \frac{a^2}{3}$. Setting $\frac{a^2}{3} = \frac{2}{n_{in}}$ yields $a = \sqrt{\frac{6}{n_{in}}}$. Thus, weights are sampled from $\mathcal{U}\left(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}}\right)$.

### Q3: How does PyTorch initialize linear layers by default?
**Answer**:
PyTorch's `nn.Linear` uses Kaiming Uniform with $a=\sqrt{5}$ (an empirical compromise: $\mathcal{U}(-\frac{1}{\sqrt{n_{in}}}, \frac{1}{\sqrt{n_{in}}})$).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Rule of thumb: ReLU $\\to$ He initialization; Tanh/Sigmoid $\\to$ Xavier initialization.
- He Normal standard deviation: $\\sigma = \\sqrt{2 / n_{in}}$.
- Xavier Normal standard deviation: $\\sigma = \\sqrt{2 / (n_{in} + n_{out})}$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=nwVOSgcrbQI)
- [Lecture Video](https://www.youtube.com/watch?v=nwVOSgcrbQI)
- [Colab Notebook](https://colab.research.google.com/drive/1Z3pWYFWgUKP7htokOj201APi574a3-vY?usp=sharing)

---
