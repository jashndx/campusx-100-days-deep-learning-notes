# Lecture 047: Backpropagation in CNN | Part 1 | Mathematical Setup

> **CampusX 100 Days of Deep Learning** | Video ID: `RvCCFttGFMY` | Duration: 32m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=RvCCFttGFMY) | **Transcript Status**: Available (hi, 3264 words)

---

## 1. Executive Summary & Core Intuition
How does backpropagation work when parameters are shared across hundreds of spatial locations?

In an ANN, each weight $w$ connects exactly one input to one output, so its gradient is simply $\delta \cdot x$.

In a CNN, a single kernel weight $K(m, n)$ is reused to compute every single pixel in the output feature map!

By the Multivariable Chain Rule, the gradient of the loss with respect to a shared parameter is the **sum of gradients over all spatial positions where that parameter was applied**.

Computing the gradient of the loss with respect to the kernel turns out to be a convolution between the input feature map and the incoming error gradient tensor $\boldsymbol{\delta}$!


## 2. Key Definitions & Formal Terminology
- **Convolutional Backpropagation**: The derivation and computation of loss gradients with respect to convolutional kernel weights, biases, and input activations.
- **Parameter Sharing Gradient Accumulation**: The mathematical rule that gradients for shared weights are computed by accumulating partial derivatives across all spatial receptive fields where the weight was utilized.
- **Error Tensor ($\boldsymbol{\delta}^{[l]}$)**: The tensor of partial derivatives of the scalar loss with respect to the pre-activation feature maps: $\\delta_{i, j} = \\frac{\\partial \\mathcal{L}}{\\partial z_{i, j}}$.


## 3. Mathematical Formulations & Derivations
**Kernel Weight Gradient Derivation:**



Forward pass for single output pixel $z_{i, j}$:

$$z_{i, j} = \sum_{m} \sum_{n} x_{i+m, j+n} K_{m, n} + b$$



The loss derivative with respect to a specific kernel weight $K_{p, q}$:

$$\frac{\partial \mathcal{L}}{\partial K_{p, q}} = \sum_{i} \sum_{j} \frac{\partial \mathcal{L}}{\partial z_{i, j}} \frac{\partial z_{i, j}}{\partial K_{p, q}}$$



Since $\frac{\partial z_{i, j}}{\partial K_{p, q}} = x_{i+p, j+q}$:

$$\frac{\partial \mathcal{L}}{\partial K_{p, q}} = \sum_{i} \sum_{j} \delta_{i, j} x_{i+p, j+q}$$



**Matrix Form:**

The gradient with respect to kernel $\mathbf{K}$ is the cross-correlation between the input activation $\mathbf{X}$ and the incoming upstream error gradient $\boldsymbol{\delta}$:

$$\nabla_{\mathbf{K}} \mathcal{L} = \mathbf{X} * \boldsymbol{\delta}$$



**Bias Gradient:**

Since bias $b$ is added to every output pixel:

$$\frac{\partial \mathcal{L}}{\partial b} = \sum_{i} \sum_{j} \delta_{i, j}$$

The bias gradient is simply the sum of all elements in the error tensor $\boldsymbol{\delta}$!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Gradient Flow Computational Diagram:**

```

Forward:

Input X (4x4)  *  Kernel K (3x3)  =======> Output Feature Map Z (2x2)



Backward:

Input X (4x4)  *  Delta dL/dZ (2x2) ======> Kernel Gradient dL/dK (3x3)

```


## 5. Implementation Code Snippet
```python

import numpy as np



# Conv2D Backward with respect to Kernel and Bias

def conv2d_backward_weights(X, dZ, K_shape):

    Kh, Kw = K_shape

    dK = np.zeros(K_shape)

    out_h, out_w = dZ.shape

    

    # Cross-correlation between X and dZ

    for i in range(Kh):

        for j in range(Kw):

            # Sum over all positions where K[i,j] contributed

            receptive_patch = X[i:i+out_h, j:j+out_w]

            dK[i, j] = np.sum(receptive_patch * dZ)

            

    db = np.sum(dZ)

    return dK, db

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why must gradients be summed over all spatial locations to compute $\\frac{\\partial \\mathcal{L}}{\\partial K}$?
**Answer**:
Because of parameter sharing: the exact same kernel weight $K_{m,n}$ was reused in computing every single output pixel $z_{i,j}$. According to the multivariable chain rule, whenever a variable influences an outcome through multiple pathways, its total derivative is the summation of derivatives across all those pathways.

### Q2: What is the spatial dimension of $\\nabla_{\\mathbf{K}} \\mathcal{L}$?
**Answer**:
It must match the exact spatial dimension of the kernel $\\mathbf{K}$ ($K_h \\times K_w \\times C_{in}$), ensuring valid subtraction during gradient descent: $\\mathbf{K} \\leftarrow \\mathbf{K} - \\eta \\nabla_{\\mathbf{K}} \\mathcal{L}$.

### Q3: How does computing $\\frac{\\partial \\mathcal{L}}{\\partial b}$ in CNN compare to ANN?
**Answer**:
In an ANN, a bias is added to a single neuron, so $\\frac{\\partial \\mathcal{L}}{\\partial b} = \\delta$. In a CNN, one scalar bias is shared across the entire 2D feature map, so its gradient is the sum of all deltas in that feature map: $\\sum_{i,j} \\delta_{i,j}$.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Kernel gradient is cross-correlation of Input with Delta: $\\nabla_K \\mathcal{L} = X * \\delta$.
- Bias gradient is the sum of all elements in the delta map.
- Summation occurs because the kernel is shared spatially.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=RvCCFttGFMY)
- [Lecture Video](https://www.youtube.com/watch?v=RvCCFttGFMY)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
