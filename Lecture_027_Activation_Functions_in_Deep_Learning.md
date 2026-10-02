# Lecture 027: Activation Functions in Deep Learning | Sigmoid, Tanh, ReLU

> **CampusX 100 Days of Deep Learning** | Video ID: `7LcUkgzx3AY` | Duration: 35m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=7LcUkgzx3AY) | **Transcript Status**: Available (hi, 6649 words)

---

## 1. Executive Summary & Core Intuition
Activation functions introduce non-linearity into neural networks, transforming them from simple linear regressors into universal function approximators.

This lecture contrasts the three foundational activation functions:

1. **Sigmoid:** Maps $(-\infty, \infty) \to (0, 1)$. Historically popular, but suffers from two fatal flaws: saturating gradients at extreme values (vanishing gradient) and non-zero-centered outputs (causing zig-zag gradient updates).

2. **Tanh (Hyperbolic Tangent):** Maps $(-\infty, \infty) \to (-1, 1)$. Solves the zero-centered issue of Sigmoid, but still suffers from gradient saturation when $|z|$ is large.

3. **ReLU (Rectified Linear Unit):** $f(z) = \max(0, z)$. The revolution that enabled deep networks: computationally trivial to evaluate, non-saturating in the positive domain ($f'(z)=1$), providing uninterrupted gradient flow. However, it suffers from the 'Dying ReLU' problem when activations become permanently negative.


## 2. Key Definitions & Formal Terminology
- **Non-Zero-Centered Outputs**: A condition where all activation outputs from a layer are strictly positive ($a > 0$), forcing all downstream weight gradients $\\frac{\\partial \\mathcal{L}}{\\partial w_i} = \\delta \\cdot a_i$ to share the exact same sign as $\\delta$, restricting gradient updates to all-positive or all-negative directions (zig-zag dynamics).
- **Saturation**: A regime where the derivative of an activation function approaches zero ($\lim_{|z| \to \infty} f'(z) = 0$), halting backpropagation gradient flow.
- **Dying ReLU Problem**: A pathological state where a neuron's weights are updated such that its pre-activation $z = \mathbf{w}^T \mathbf{x} + b < 0$ for all training inputs, resulting in zero output and zero gradient, permanently disabling the neuron.


## 3. Mathematical Formulations & Derivations
**Mathematical Comparison of Core Activations:**



1. **Sigmoid:**

   $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z))$$

   $$\text{Range: } (0, 1), \quad \max(\sigma') = 0.25 \text{ at } z=0$$



2. **Tanh:**

   $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \quad \tanh'(z) = 1 - \tanh^2(z)$$

   $$\text{Range: } (-1, 1), \quad \max(\tanh') = 1.0 \text{ at } z=0$$



3. **ReLU:**

   $$f(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}$$

   $$f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases} \quad (\text{Subgradient at } z=0: [0, 1])$$

   $$\text{Range: } [0, \infty), \quad \text{Saturation: Only for } z \le 0$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Comparison Matrix:**



| Property | Sigmoid | Tanh | ReLU |

| :--- | :--- | :--- | :--- |

| **Output Range** | $(0, 1)$ | $(-1, 1)$ | $[0, \infty)$ |

| **Zero-Centered?** | No | Yes | No |

| **Vanishing Gradient?**| Severe ($max = 0.25$) | Moderate ($max = 1.0$, saturates) | None for $z > 0$ ($deriv = 1$) |

| **Dying Neuron Risk?** | No | No | Yes (Dying ReLU) |

| **Compute Cost** | High (Exponential $e^{-z}$) | High (Exponential) | Ultra-Fast ($\max(0, z)$) |

| **Primary Modern Use** | Binary Output Layer | RNN Hidden States | General Hidden Layers |


## 5. Implementation Code Snippet
```python

import numpy as np



def sigmoid(z):

    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))



def tanh(z):

    return np.tanh(z)



def relu(z):

    return np.maximum(0, z)



def relu_derivative(z):

    return (z > 0).astype(float)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does non-zero-centered activation (like Sigmoid) cause zig-zag gradient updates?
**Answer**:
Since $\\frac{\\partial \\mathcal{L}}{\\partial w_i} = \\delta \\cdot a_i$, if all $a_i > 0$, the signs of the gradients for all weights in that layer are entirely dictated by the scalar sign of $\\delta$. Consequently, all weights connected to that neuron must either simultaneously increase or simultaneously decrease. If the true optimal trajectory requires some weights to increase and others to decrease, the optimizer is forced into an inefficient zig-zag path.

### Q2: Why is Tanh preferred over Sigmoid in hidden layers?
**Answer**:
Tanh is strictly zero-centered (mean activation is close to 0), which eliminates the systematic zig-zag gradient updates of Sigmoid. Furthermore, its maximum derivative is $1.0$ (four times larger than Sigmoid's $0.25$), providing stronger gradient flow.

### Q3: What causes the 'Dying ReLU' problem and how is it fixed in practice?
**Answer**:
If an aggressive learning rate takes an excessively large step, weights can update such that the neuron outputs negative values for all samples in the dataset. Because the derivative is zero for all negative values, no gradient ever flows back to update the weights again. Solutions include using Leaky ReLU, lower learning rates, or He weight initialization.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Use ReLU for hidden layers by default.
- Use Sigmoid only for binary classification output layer.
- Tanh is zero-centered; Sigmoid is not.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=7LcUkgzx3AY)
- [Lecture Video](https://www.youtube.com/watch?v=7LcUkgzx3AY)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
