# Lecture 048: CNN Backpropagation Part 2 | Gradients in MaxPool, Flatten & Conv

> **CampusX 100 Days of Deep Learning** | Video ID: `OoSDzOodY3Y` | Duration: 35m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=OoSDzOodY3Y) | **Transcript Status**: Available (hi, 3302 words)

---

## 1. Executive Summary & Core Intuition
This lecture completes the backpropagation derivation through all CNN layer types:

1. **Backprop through MaxPooling:** Max pooling has no weights, but must transmit errors backward to previous layers. Gradients route *exclusively* to the specific spatial index that achieved the maximum during the forward pass (using a cached binary boolean mask); all other positions receive zero.

2. **Backprop through Flatten:** Reshapes the 1D gradient vector back into the identical 3D tensor shape $(H, W, C)$ of the preceding convolutional feature maps.

3. **Backprop to Input Feature Map ($\frac{\partial \mathcal{L}}{\partial \mathbf{X}}$):** To pass gradients to earlier layers, we compute the gradient with respect to $\mathbf{X}$. Mathematically, this equals the 'Full Convolution' of the upstream error $\boldsymbol{\delta}$ with the spatially **flipped (rotated 180 degrees) kernel** $\mathbf{K}^{rot180}$!


## 2. Key Definitions & Formal Terminology
- **Argmax Switch / Mask**: A binary matrix recorded during the forward pass of MaxPooling that marks the spatial coordinates of maximal values with 1 and all other values with 0.
- **Full Convolution**: A convolution where sufficient zero-padding is added such that the output dimension is larger than the input: $N_{out} = N_{in} + K - 1$.
- **180-Degree Rotated Kernel ($\mathbf{K}^{rot180}$)**: A kernel whose rows and columns are flipped: $K^{rot180}(m, n) = K(Kh - 1 - m, Kw - 1 - n)$, which arises naturally from reversing index summation in the chain rule.


## 3. Mathematical Formulations & Derivations
**1. Backpropagation through MaxPooling:**

Let mask $M(i, j) = 1$ if $x(i, j) = \max(\text{patch})$ else $0$:

$$\frac{\partial \mathcal{L}}{\partial x(i, j)} = \delta_{out} \cdot M(i, j)$$



**2. Backpropagation with respect to Input Activations $\mathbf{X}$:**

To propagate error $\boldsymbol{\delta}^{[l-1]} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}}$ to the previous layer:



$$\frac{\partial \mathcal{L}}{\partial x_{i, j}} = \sum_{m} \sum_{n} \delta_{i-m, j-n} K_{m, n} = \boldsymbol{\delta} *_{\text{full}} \mathbf{K}^{rot180}$$



Where $\mathbf{K}^{rot180}$ is the kernel rotated by $180^\circ$, and $*_{\text{full}}$ denotes convolution with padding $P = K - 1$.

Output dimension matches input $\mathbf{X}$:

$$M_{out} = M_\delta + K - 1 = (N - K + 1) + K - 1 = N$$

Dimensions match input $\mathbf{X}$!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**MaxPooling Gradient Routing:**

```

Forward Pass:                       Backward Pass:

[ 1   4 ] -> MaxPool -> [ 4 ]      [ 0   dL/dy ] <- Route <- [ dL/dy ]

[ 2   3 ]   (Index: top-right)     [ 0     0   ]

```

All non-maximal elements receive zero gradient.


## 5. Implementation Code Snippet
```python

import numpy as np



# Backprop through MaxPooling

def maxpool_backward(dZ, X, pool_size=2, stride=2):

    dX = np.zeros_like(X)

    out_h, out_w = dZ.shape

    

    for i in range(out_h):

        for j in range(out_w):

            h_start = i * stride

            h_end = h_start + pool_size

            w_start = j * stride

            w_end = w_start + pool_size

            

            patch = X[h_start:h_end, w_start:w_end]

            max_val = np.max(patch)

            # Binary mask: 1 at max location, 0 elsewhere

            mask = (patch == max_val)

            dX[h_start:h_end, w_start:w_end] += mask * dZ[i, j]

            

    return dX

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does the kernel need to be rotated 180 degrees when computing $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{X}}$?
**Answer**:
In the forward pass, moving right on the image moves forward across the kernel. To trace which input pixel contributed to which output gradient, the relative directional displacement is reversed: an input pixel to the right of another contributes to output pixels to the left. Reversing this index mapping in the chain rule inversion is mathematically equivalent to rotating the kernel by $180^\circ$.

### Q2: What happens during backpropagation if two elements in a MaxPooling patch share the exact same maximum value?
**Answer**:
In standard subgradient convention, the upstream gradient is either split equally between the tied elements (divided by 2) or assigned to the first index encountered by `argmax`.

### Q3: What is the operation of the Flatten backward pass?
**Answer**:
A simple tensor reshape: `dFlatten.reshape(conv_output_shape)`. It has zero floating-point operations and zero trainable parameters.

## 7. Crucial Exam Takeaways & Common Pitfalls
- MaxPool backward pass routes gradient ONLY to the argmax index.
- Input gradient is Full Convolution with 180-degree flipped kernel: $\\delta *_{full} K^{rot180}$.
- Flatten backward pass is a pure tensor reshape.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=OoSDzOodY3Y)
- [Lecture Video](https://www.youtube.com/watch?v=OoSDzOodY3Y)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
