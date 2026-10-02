# Lecture 028: ReLU Variants Explained | Leaky ReLU, PReLU, ELU, SELU

> **CampusX 100 Days of Deep Learning** | Video ID: `2OwWs7Hzr9g` | Duration: 31m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=2OwWs7Hzr9g) | **Transcript Status**: Available (hi, 4577 words)

---

## 1. Executive Summary & Core Intuition
While standard ReLU transformed deep learning, its 'Dying ReLU' flaw (where up to 40% of neurons in a trained network can become permanently inactive) motivated the development of advanced variants.

This lecture details:

1. **Leaky ReLU:** Assigns a small constant slope $\alpha$ (typically $0.01$) to negative inputs, ensuring a non-zero gradient always exists.

2. **Parametric ReLU (PReLU):** Turns the slope $\alpha$ into a learnable parameter trained via backpropagation.

3. **Exponential Linear Unit (ELU):** Uses an exponential curve for negative inputs, bringing mean activations closer to zero while smoothly saturating for noise robustness.

4. **Scaled ELU (SELU):** Incorporates self-normalizing properties: under specific conditions, activations automatically preserve zero mean and unit variance across arbitrary depth without Batch Normalization!


## 2. Key Definitions & Formal Terminology
- **Leaky ReLU**: A ReLU variant with a small non-zero slope for $z < 0$: $f(z) = \max(\alpha z, z)$ where $\alpha \approx 0.01$.
- **Parametric ReLU (PReLU)**: A generalized Leaky ReLU where the negative slope $\alpha$ is a trainable weight updated via gradient descent.
- **Exponential Linear Unit (ELU)**: An activation function featuring smooth exponential transitions for negative values: $f(z) = \alpha(e^z - 1)$ for $z \le 0$.
- **Self-Normalizing Neural Network (SNN)**: A neural network constructed with SELU activations and LeCun normal initialization that automatically maintains stable variance and zero mean across arbitrarily deep architectures.


## 3. Mathematical Formulations & Derivations
**Mathematical Definitions of ReLU Variants:**



1. **Leaky ReLU ($\alpha = 0.01$):**

   $$f(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha z & \text{if } z \le 0 \end{cases}, \quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ \alpha & \text{if } z < 0 \end{cases}$$



2. **Parametric ReLU (PReLU):**

   $$f(z) = \max(\alpha z, z), \quad \frac{\partial \mathcal{L}}{\partial \alpha} = \sum_{z < 0} \delta \cdot z$$



3. **ELU ($\alpha > 0$):**

   $$f(z) = \begin{cases} z & \text{if } z > 0 \\ \alpha (e^z - 1) & \text{if } z \le 0 \end{cases}, \quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ f(z) + \alpha & \text{if } z \le 0 \end{cases}$$



4. **SELU ($\lambda \approx 1.0507, \ \alpha \approx 1.6733$):**

   $$f(z) = \lambda \begin{cases} z & \text{if } z > 0 \\ \alpha(e^z - 1) & \text{if } z \le 0 \end{cases}$$

   $\lambda$ and $\alpha$ are analytically derived fixed points of Banach's fixed-point theorem that preserve $\mathbb{E}[a]=0, \text{Var}(a)=1$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Summary of Variants Trade-Offs:**



| Variant | Equation ($z \le 0$) | Solves Dying ReLU? | Extra Parameters | Differentiable at 0? |

| :--- | :--- | :--- | :--- | :--- |

| **Standard ReLU** | $0$ | No | 0 | No (subgradient) |

| **Leaky ReLU** | $0.01 z$ | Yes | 0 | No |

| **PReLU** | $\alpha z$ | Yes | $1$ per layer/channel | No |

| **ELU** | $\alpha(e^z - 1)$ | Yes | 0 | Yes (if $\alpha=1$) |

| **SELU** | $\lambda \alpha (e^z - 1)$ | Yes | 0 | Yes |


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Implementing ReLU variants in Keras

model = models.Sequential([

    layers.Dense(64, input_shape=(30,)),

    layers.LeakyReLU(alpha=0.01), # Leaky ReLU as a layer

    

    layers.Dense(64),

    layers.PReLU(),               # Learnable alpha slope

    

    layers.Dense(64, activation='elu'),

    

    # Self-Normalizing Layer (Must use lecun_normal initialization!)

    layers.Dense(32, activation='selu', kernel_initializer='lecun_normal'),

    layers.Dense(1, activation='sigmoid')

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is ELU computationally more expensive than Leaky ReLU?
**Answer**:
ELU computes the transcendental exponential function $e^z$ for all negative inputs. In high-throughput training pipelines, exponential evaluations are significantly slower than the single scalar multiplication ($\alpha \cdot z$) of Leaky ReLU.

### Q2: What strict prerequisites must be satisfied for SELU to be self-normalizing?
**Answer**:
Klambauer et al. (2017) proved that SELU is self-normalizing if and only if: (1) Weights are initialized with `lecun_normal`, (2) Input features are standardized ($\mu=0, \sigma=1$), (3) Architecture consists of standard dense feedforward layers without Batch Normalization, and (4) If dropout is used, `AlphaDropout` must be used instead of standard Dropout.

### Q3: How does PReLU prevent overfitting despite adding learnable parameters?
**Answer**:
PReLU introduces only a single scalar parameter $\alpha$ per layer (or per convolutional feature channel). Compared to millions of weights, adding one parameter introduces negligible overfitting risk while providing layer-adaptive flexibility.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Leaky ReLU replaces zero derivative with small slope $\\alpha = 0.01$.
- SELU requires `lecun_normal` and `AlphaDropout` to preserve self-normalization.
- ELU is smooth everywhere and zero-centered for negative values.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=2OwWs7Hzr9g)
- [Lecture Video](https://www.youtube.com/watch?v=2OwWs7Hzr9g)

---
