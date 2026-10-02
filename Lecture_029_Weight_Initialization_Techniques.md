# Lecture 029: Weight Initialization Techniques | What NOT to Do

> **CampusX 100 Days of Deep Learning** | Video ID: `2MSY0HwH5Ss` | Duration: 28m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=2MSY0HwH5Ss) | **Transcript Status**: Available (en-US, 6662 words)

---

## 1. Executive Summary & Core Intuition
How should neural network weights be initialized prior to training?

This lecture explores the catastrophic failure modes of naive initialization:

1. **Zero Initialization ($\mathbf{W} = \mathbf{0}$):** All neurons in a hidden layer compute identical outputs ($z_j = b$). Consequently, all hidden neurons receive identical error gradients and undergo identical weight updates throughout training. The network exhibits complete **Symmetry** and collapses to the representational capacity of a single neuron!

2. **Constant Initialization ($\mathbf{W} = c$):** Identical to zero initialization; neurons remain symmetric.

3. **Large Random Initialization ($\mathbf{W} \sim \mathcal{N}(0, 1)$):** When inputs pass through multiple layers, the variance of pre-activations explodes proportionally to $\prod \text{fan\_in}$. Activations saturate immediately, triggering catastrophic vanishing gradients in Sigmoid/Tanh and exploding gradients in ReLU.

4. **Too Small Random Initialization ($\mathbf{W} \sim \mathcal{N}(0, 0.001)$):** Activation variance shrinks exponentially with depth, collapsing activations to zero.


## 2. Key Definitions & Formal Terminology
- **Symmetry Problem (Symmetry Breaking)**: A condition where all neurons in a layer compute identical functions and receive identical updates because weights were initialized symmetrically, preventing distinct feature learning.
- **Fan-in ($n_{in}$)**: The number of incoming inputs/connections feeding into a neuron.
- **Fan-out ($n_{out}$)**: The number of output connections departing from a neuron to the subsequent layer.
- **Variance Preservation**: The mathematical principle that the variance of activations in the forward pass and gradients in the backward pass should remain constant across all layers.


## 3. Mathematical Formulations & Derivations
**Proof of Symmetry under Zero Initialization:**

Let $\mathbf{W}^{[1]} = \mathbf{0}, \mathbf{b}^{[1]} = \mathbf{0}$.

Forward pass for any neuron $j$:

$$z_j^{[1]} = \sum_k w_{jk}^{[1]} x_k + b_j^{[1]} = 0 \implies a_j^{[1]} = g(0)$$

For all neurons $j \in \{1, \dots, n^{[1]}\}$, $a_j^{[1]}$ is identical!



Backward pass error signal:

$$\delta_j^{[1]} = \left( \sum_p w_{pj}^{[2]} \delta_p^{[2]} \right) g'(z_j^{[1]})$$

If $\mathbf{W}^{[2]}$ is also symmetric:

$$\frac{\partial \mathcal{L}}{\partial w_{jk}^{[1]}} = \delta_j^{[1]} x_k = \frac{\partial \mathcal{L}}{\partial w_{mk}^{[1]}} \quad \forall j, m$$

All neurons receive the exact same gradient update:

$$w_{jk}^{[1](t+1)} = w_{mk}^{[1](t+1)}$$

Neurons remain symmetric forever.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Variance Dynamics through Linear Transformation:**

Let $z = \sum_{i=1}^{n_{in}} w_i x_i$. Assuming $w_i$ and $x_i$ are independent with mean zero:

$$\text{Var}(z) = \sum_{i=1}^{n_{in}} \text{Var}(w_i x_i) = \sum_{i=1}^{n_{in}} \left[ \mathbb{E}[w_i]^2 \text{Var}(x_i) + \mathbb{E}[x_i]^2 \text{Var}(w_i) + \text{Var}(w_i)\text{Var}(x_i) \right]$$

Since $\mathbb{E}[w_i] = 0$ and $\mathbb{E}[x_i] = 0$:

$$\text{Var}(z) = n_{in} \cdot \text{Var}(w) \cdot \text{Var}(x)$$



**Fundamental Law of Initialization:**

To maintain stable signal propagation ($\text{Var}(z) = \text{Var}(x)$), we must have:

$$\text{Var}(w) = \frac{1}{n_{in}}$$


## 5. Implementation Code Snippet
```python

import numpy as np

import matplotlib.pyplot as plt



# Simulating signal propagation across 10 layers with large weights

D = np.random.randn(1000, 500) # Input: 1000 samples, 500 features

layers = [500] * 10

activations = []



curr = D

for fan_in in layers:

    # BAD: Large random weights (std = 1.0)

    W = np.random.randn(fan_in, fan_in) * 1.0 

    curr = np.tanh(np.dot(curr, W))

    activations.append(curr)



print(f"Layer 1 std: {activations[0].std():.4f}")

print(f"Layer 10 std: {activations[-1].std():.4f}")

# Observes activations collapsing to -1.0 or +1.0 (Saturation!)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why can biases be initialized to zero even though weights cannot?
**Answer**:
Symmetry breaking is achieved entirely by the weights being randomized. If weights are distinct random numbers, neurons receive distinct inputs and compute distinct outputs. Setting biases to zero ($b=0$) is safe and standard practice.

### Q2: What happens if you initialize weights from a uniform distribution $\mathcal{U}(-1, 1)$ in a deep network?
**Answer**:
The variance of a uniform distribution $\mathcal{U}(-a, a)$ is $\frac{a^2}{3}$. For $a=1$, variance is $\frac{1}{3} \approx 0.33$. If $n_{in} = 100$, then $\text{Var}(z) = 100 \times 0.33 \times \text{Var}(x) = 33 \text{Var}(x)$. Variance will explode exponentially with depth, saturating all neurons.

### Q3: Why did early deep networks in the 1990s and 2000s fail to train properly?
**Answer**:
Researchers initialized weights using standard Gaussian distributions $\mathcal{N}(0, 1)$ or tiny constants, unaware that variance scales with $n_{in}$, resulting in immediate signal decay or saturation before training even started.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Never initialize weights to zero or a constant (Symmetry problem).
- Variance rule: $\\text{Var}(z) = n_{in} \\cdot \\text{Var}(w) \\cdot \\text{Var}(x)$.
- Biases CAN and should be initialized to zero.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=2MSY0HwH5Ss)
- [Lecture Video](https://www.youtube.com/watch?v=2MSY0HwH5Ss)
- [Colab Notebook](https://colab.research.google.com/drive/1M4q5yRA0iQXh9h8Y3J7zFGIQzO_Pv9n0?usp=sharing)

---
