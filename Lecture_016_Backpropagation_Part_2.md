# Lecture 016: Backpropagation Part 2 | The How | Mathematical Derivation

> **CampusX 100 Days of Deep Learning** | Video ID: `ma6hWrU-LaI` | Duration: 45m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=ma6hWrU-LaI) | **Transcript Status**: Available (en-US, 7796 words)

---

## 1. Executive Summary & Core Intuition
This lecture delivers the step-by-step rigorous mathematical derivation of the Four Fundamental Equations of Backpropagation.

Tracing a complete 2-layer network with 2 inputs, 2 hidden neurons, and 1 output neuron under MSE loss, we derive:

1. Output layer error $\delta^{[L]}$.

2. Hidden layer error $\delta^{[l]}$ in terms of the next layer's error $\delta^{[l+1]}$.

3. Rate of change of cost with respect to any bias.

4. Rate of change of cost with respect to any weight.

This establishes the recursive formula that is vectorized across entire layers and mini-batches.


## 2. Key Definitions & Formal Terminology
- **Four Fundamental Equations of Backpropagation**: The set of four matrix equations formulated by Rumelhart and Nielsen that completely specify the backward computation across any feedforward neural network.
- **Hadamard Product ($\odot$)**: Element-wise matrix multiplication, where $(A \odot B)_{ij} = A_{ij} B_{ij}$.
- **Jacobian Matrix**: A matrix of all first-order partial derivatives of a vector-valued function.


## 3. Mathematical Formulations & Derivations
**The 4 Fundamental Equations of Backpropagation (Vectorized Form):**



**Equation 1 (Output Error $\boldsymbol{\delta}^{[L]}$):**

$$\boldsymbol{\delta}^{[L]} = \nabla_{\mathbf{a}} \mathcal{L} \odot g'(\mathbf{z}^{[L]})$$

*For MSE loss and Sigmoid activation:*

$$\boldsymbol{\delta}^{[L]} = (\mathbf{a}^{[L]} - \mathbf{y}) \odot \mathbf{a}^{[L]} \odot (1 - \mathbf{a}^{[L]})$$

*For Cross-Entropy loss and Softmax/Sigmoid activation:*

$$\boldsymbol{\delta}^{[L]} = \mathbf{a}^{[L]} - \mathbf{y}$$



**Equation 2 (Hidden Layer Error Propagation $\boldsymbol{\delta}^{[l]}$):**

$$\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'(\mathbf{z}^{[l]})$$



**Equation 3 (Gradient with Respect to Biases $\nabla_{\mathbf{b}^{[l]}} \mathcal{L}$):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]} \quad \left(\text{For batch: } \frac{1}{M} \sum_{i=1}^M \boldsymbol{\delta}^{[l](i)} \right)$$



**Equation 4 (Gradient with Respect to Weights $\nabla_{\mathbf{W}^{[l]}} \mathcal{L}$):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T \quad \left(\text{For batch: } \frac{1}{M} \mathbf{A}^{[l-1] T} \boldsymbol{\Delta}^{[l]} \right)$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Complete Vectorized Backpropagation Algorithm:**

1. **Initialize** backward pass with loss derivative: $\frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[L]}}$.

2. **Compute Output Error:** $\boldsymbol{\Delta}^{[L]} = \frac{\partial \mathcal{L}}{\partial \mathbf{A}^{[L]}} \odot g'(\mathbf{Z}^{[L]})$.

3. **Loop backwards** from $l = L, L-1, \dots, 1$:

   a. Compute weight gradients: $d\mathbf{W}^{[l]} = \frac{1}{M} (\mathbf{A}^{[l-1]})^T \boldsymbol{\Delta}^{[l]}$.

   b. Compute bias gradients: $d\mathbf{b}^{[l]} = \frac{1}{M} \sum_{rows} \boldsymbol{\Delta}^{[l]}$.

   c. If $l > 1$, propagate error to previous layer:

      $$\boldsymbol{\Delta}^{[l-1]} = (\boldsymbol{\Delta}^{[l]} (\mathbf{W}^{[l]})^T) \odot g'(\mathbf{Z}^{[l-1]})$$

4. **Update Parameters:**

   $$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \, d\mathbf{W}^{[l]}$$

   $$\mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \, d\mathbf{b}^{[l]}$$


## 5. Implementation Code Snippet
```python

import numpy as np



def backward_propagation(AL, Y, caches):

    grads = {}

    L = len(caches) # number of layers

    M = AL.shape[0] # batch size

    Y = Y.reshape(AL.shape)

    

    # 1. Output layer error (BCE + Sigmoid shortcut: AL - Y)

    dZL = AL - Y

    A_prev, W, b, Z = caches[-1]

    grads[f"dW{L}"] = (1 / M) * np.dot(A_prev.T, dZL)

    grads[f"db{L}"] = (1 / M) * np.sum(dZL, axis=0, keepdims=True)

    dZ_curr = dZL

    

    # 2. Loop backwards

    for l in reversed(range(1, L)):

        A_prev, W_curr, b_curr, Z_curr = caches[l-1]

        _, W_next, _, _ = caches[l]

        

        # ReLU derivative: 1 if Z > 0 else 0

        dZ_prev = np.dot(dZ_curr, W_next.T) * (Z_curr > 0)

        grads[f"dW{l}"] = (1 / M) * np.dot(A_prev.T, dZ_prev)

        grads[f"db{l}"] = (1 / M) * np.sum(dZ_prev, axis=0, keepdims=True)

        dZ_curr = dZ_prev

        

    return grads

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Derive the output layer error $\\delta^{[L]}$ for a network using Cross-Entropy loss and Softmax activation.
**Answer**:
For Softmax $\\hat{y}_k = \\frac{e^{z_k}}{\\sum e^{z_j}}$ and Cross-Entropy $\\mathcal{L} = -\\sum y_i \\ln \\hat{y}_i$: using the chain rule $\\frac{\\partial \\mathcal{L}}{\\partial z_k} = \\sum_j \\frac{\\partial \\mathcal{L}}{\\partial \\hat{y}_j} \\frac{\\partial \\hat{y}_j}{\\partial z_k}$. Since $\\frac{\\partial \\hat{y}_j}{\\partial z_k} = \\hat{y}_k(\\delta_{jk} - \\hat{y}_j)$, substituting and simplifying yields the elegant result: $\\delta_k^{[L]} = \\hat{y}_k - y_k$.

### Q2: Why does error propagation involve the transpose of the weight matrix $(\\mathbf{W}^{[l+1]})^T$?
**Answer**:
In the forward pass, layer $l$ outputs map to layer $l+1$ via $\\mathbf{A}^{[l]} \\mathbf{W}^{[l+1]}$. In the backward pass, gradients flow in reverse from $l+1$ to $l$. To match dimensional projections, the weight matrix must be transposed: $(\\text{dim}_{l+1}) \\times (\\text{dim}_{l+1} \\times \\text{dim}_l)^T = \\text{dim}_l$.

### Q3: What is the computational bottleneck in backpropagation?
**Answer**:
The large matrix multiplications $\\mathbf{A}^T \\boldsymbol{\\Delta}$ and $\\boldsymbol{\\Delta} \\mathbf{W}^T$ at each layer, which are compute-bound and scale as $\\mathcal{O}(M \\cdot n_{in} \\cdot n_{out})$ operations.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Output error under BCE+Sigmoid or CCE+Softmax is always: $\\hat{y} - y$.
- Hidden error equation: $\\boldsymbol{\\delta}^{[l]} = ((\\mathbf{W}^{[l+1]})^T \\boldsymbol{\\delta}^{[l+1]}) \\odot g'(\\mathbf{Z}^{[l]})$.
- Transposing weights reverses the directional flow of dimensions.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=ma6hWrU-LaI)
- [Lecture Video](https://www.youtube.com/watch?v=ma6hWrU-LaI)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
