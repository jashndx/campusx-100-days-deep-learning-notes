# Lecture 060: Problems with RNN | Vanishing Gradients & Long-Term Forgetfulness

> **CampusX 100 Days of Deep Learning** | Video ID: `AWHSZzp96kM` | Duration: 28m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=AWHSZzp96kM) | **Transcript Status**: Available (en-US, 3740 words)

---

## 1. Executive Summary & Core Intuition
Why did Vanilla RNNs fail in real-world NLP and sequential tasks before the advent of LSTMs?

This lecture analyzes the two fatal pathologies of Vanilla RNNs:

1. **Vanishing Gradient Pathology:** As derived in BPTT, transmitting error gradients across $T$ steps requires multiplying by $\prod_{k} \mathbf{W}_{hh}^T \text{diag}(1 - \mathbf{h}_k^2)$. 

   Since $\tanh' \le 1.0$ and $\mathbf{W}_{hh}$ eigenvalues are typically $< 1$, the gradient decays to absolute zero after just 10 to 15 time steps!

   Consequently, words at the beginning of a sentence have zero influence on weight updates. The network develops **severe amnesia / short-term memory**.

   - Example: In the sentence "The **clouds** in the sky are ... **white**", the distance is short (3 words); RNN succeeds.

   - Example: In "I grew up in **France**, spoke fluent Spanish, traveled ... and I speak fluent **French**", the distance is 40 words; the RNN cannot connect "France" to "French"!

2. **Exploding Gradient Pathology:** If $\mathbf{W}_{hh}$ eigenvalues exceed $1.0$, gradients explode exponentially into `NaN`, causing parameter values to jump uncontrollably.


## 2. Key Definitions & Formal Terminology
- **Long-Term Dependency Problem**: The inability of vanilla RNNs to learn relationships between tokens that are separated by more than 10-15 time steps due to vanishing gradients.
- **Spectral Radius ($\rho(\mathbf{W})$)**: The maximum absolute value of the eigenvalues of a matrix: $\\rho(\\mathbf{W}) = \\max_i |\\lambda_i|$. If $\\rho(\\mathbf{W}_{hh}) < 1$, gradients vanish; if $\\rho(\\mathbf{W}_{hh}) > 1$, gradients explode.
- **Gradient Clipping by Norm**: The standard remedy for exploding gradients in RNNs: if $\\|\\mathbf{g}\\| > c$, set $\\mathbf{g} \\leftarrow c \\frac{\\mathbf{g}}{\\|\\mathbf{g}\\|}$.


## 3. Mathematical Formulations & Derivations
**Pascanu, Mikolov & Bengio's Theorem on Vanishing Gradients (2013):**



Let the temporal Jacobian be $\mathbf{J}_k = \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \mathbf{W}_{hh}^T \text{diag}(1 - \mathbf{h}_k^2)$.

The norm of the gradient signal propagating from step $t$ back to step $j$ satisfies:



$$\left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \right\| \le \left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \right\| \cdot \prod_{k=j+1}^t \|\mathbf{J}_k\| \le \left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \right\| \cdot (\gamma)^{t-j}$$



Where $\gamma = \lambda_{\max}(\mathbf{W}_{hh}) \cdot \max|\tanh'| = \lambda_{\max}(\mathbf{W}_{hh}) \cdot 1.0$.

- **Case 1 ($\gamma < 1$):** $\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \| \to 0$ exponentially as $(t - j) \to \infty$ (**Vanishing Gradient**).

- **Case 2 ($\gamma > 1$):** $\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \| \to \infty$ exponentially as $(t - j) \to \infty$ (**Exploding Gradient**).


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Gradient Magnitude Decay over Time Steps:**

```

Gradient Signal

 ^

 | 1.0  (Step t)

 |  *

 |   \

 |    \ 0.1 (Step t - 5)

 |     \

 |      *_________ 0.00001 (Step t - 15)  <-- Completely vanishes!

 0----------------------------------------> Backward Steps (t - j)

```

Vanilla RNNs are practically incapable of bridging temporal spans $> 15$ steps.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models, optimizers



# Fixing exploding gradients in RNNs via Gradient Clipping

model = models.Sequential([

    layers.SimpleRNN(64, input_shape=(None, 50)),

    layers.Dense(1)

])



# clipnorm=1.0 scales gradient vector if norm exceeds 1.0

custom_opt = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)

model.compile(optimizer=custom_opt, loss='mse')

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Gradient Clipping effective for exploding gradients, but INEFFECTIVE for vanishing gradients?
**Answer**:
Exploding gradients have a distinct symptom: vector norm exceeds a threshold ($\|\mathbf{g}\| > c$). Clipping truncates the vector, preserving direction and preventing numerical overflow. For vanishing gradients, the gradient has vanished to zero; you cannot multiply zero by a scalar to recover the lost directional information! Overcoming vanishing gradients requires architectural changes (LSTMs, GRUs, Residual connections).

### Q2: What is the maximum effective context memory span of a Vanilla RNN?
**Answer**:
Empirically, vanilla RNNs can reliably retain context across only $8$ to $15$ time steps. Beyond 15 steps, the gradients decay below numerical precision ($< 10^{-7}$), making it impossible to learn dependencies across paragraphs or long audio clips.

### Q3: Why does initializing $\mathbf{W}_{hh}$ as an Identity matrix (IRNN) help mitigate vanishing gradients?
**Answer**:
Le et al. (2015) showed that initializing $\mathbf{W}_{hh} = \mathbf{I}$ (the identity matrix) with ReLU activations sets initial eigenvalues to $1.0$ ($\lambda = 1$). In early epochs, activations and gradients pass unchanged through time ($\mathbf{I}^T = \mathbf{I}$), mimicking constant memory flow.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Vanilla RNN context limit: $\\approx 10-15$ steps.
- Gradient clipping solves exploding gradients, but CANNOT solve vanishing gradients.
- Vanishing gradients necessitated the invention of LSTMs and GRUs.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=AWHSZzp96kM)
- [Lecture Video](https://www.youtube.com/watch?v=AWHSZzp96kM)

---
