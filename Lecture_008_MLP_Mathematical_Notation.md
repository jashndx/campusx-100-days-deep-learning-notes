# Lecture 008: MLP Mathematical Notation | Standard Layer Indexing

> **CampusX 100 Days of Deep Learning** | Video ID: `H0_3SJh4Rqs` | Duration: 18m 35s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=H0_3SJh4Rqs) | **Transcript Status**: Available (hi, 2173 words)

---

## 1. Executive Summary & Core Intuition
Before deriving backpropagation and multi-layer networks, establishing rigorous, standardized mathematical notation is paramount.

In deep learning literature (e.g., Goodfellow, Deep Learning Book; Andrew Ng; Michael Nielsen), ambiguous index naming is the #1 source of confusion.

This lecture codifies the standard index notation:

- Layers are indexed by superscript brackets: $[l]$ denotes layer $l$. Layer $[0]$ is input, layer $[L]$ is output.

- Weight $w_{jk}^{[l]}$ denotes the weight connecting neuron $k$ in layer $[l-1]$ to neuron $j$ in layer $[l]$.

- Bias $b_j^{[l]}$ is the bias of neuron $j$ in layer $[l]$.

- Linear combination: $z_j^{[l]}$.

- Activated output: $a_j^{[l]} = \sigma(z_j^{[l]})$.


## 2. Key Definitions & Formal Terminology
- **Weight Matrix $\mathbf{W}^{[l]}$**: A 2D tensor of shape $(n^{[l]}, n^{[l-1]})$ or $(n^{[l-1]}, n^{[l]})$ whose elements govern the affine transformation from layer $l-1$ to layer $l$.
- **Activation Vector $\mathbf{A}^{[l]}$**: The vector of non-linearly transformed post-activation values emerging from layer $l$.
- **Pre-activation Vector $\mathbf{Z}^{[l]}$**: The raw linear combination vector $\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$ before applying the activation function.


## 3. Mathematical Formulations & Derivations
**Standard Deep Learning Notation Glossary:**



1. **Layer Index:**

   - $L$: Total number of layers in the network.

   - $n^{[l]}$: Number of units (neurons) in layer $l$.

   - $n^{[0]} = D_{in}$ (input dimensionality), $n^{[L]} = D_{out}$ (output classes/targets).



2. **Forward Propagation Equations (Single Sample):**

   $$z_j^{[l]} = \sum_{k=1}^{n^{[l-1]}} w_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]}$$

   $$a_j^{[l]} = g^{[l]}(z_j^{[l]})$$



3. **Matrix Vectorized Form (Batch of $M$ samples):**

   Let $\mathbf{A}^{[l-1]} \in \mathbb{R}^{M \times n^{[l-1]}}$, $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$, $\mathbf{b}^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$:

   $$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$

   $$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

   Where $\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{M \times n^{[0]}}$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Dimensionality Sanity Check Table:**



| Tensor Symbol | Description | Shape (Sample-first / Keras convention) |

| :--- | :--- | :--- |

| $\mathbf{X}$ | Input Batch | $(M, n^{[0]})$ |

| $\mathbf{W}^{[l]}$ | Weights for layer $l$ | $(n^{[l-1]}, n^{[l]})$ |

| $\mathbf{b}^{[l]}$ | Biases for layer $l$ | $(1, n^{[l]})$ (broadcasted across $M$) |

| $\mathbf{Z}^{[l]}$ | Pre-activations | $(M, n^{[l]})$ |

| $\mathbf{A}^{[l]}$ | Post-activations | $(M, n^{[l]})$ |

| $\hat{\mathbf{Y}} = \mathbf{A}^{[L]}$ | Network Output | $(M, n^{[L]})$ |


## 5. Implementation Code Snippet
```python

import numpy as np



# Verifying dimensional alignment for 3-layer MLP

M = 32        # Batch size

n_0 = 10      # Input features

n_1 = 64      # Hidden layer 1

n_2 = 32      # Hidden layer 2

n_3 = 1       # Output neuron



X = np.random.randn(M, n_0)

W1 = np.random.randn(n_0, n_1)

b1 = np.zeros((1, n_1))



W2 = np.random.randn(n_1, n_2)

b2 = np.zeros((1, n_2))



W3 = np.random.randn(n_2, n_3)

b3 = np.zeros((1, n_3))



# Forward pass step 1

Z1 = np.dot(X, W1) + b1

A1 = np.maximum(0, Z1) # ReLU

assert A1.shape == (M, n_1)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: In the notation $w_{jk}^{[l]}$, what do $j, k,$ and $l$ denote?
**Answer**:
$l$ denotes the target layer index. $j$ denotes the destination neuron index in layer $l$. $k$ denotes the source neuron index in the preceding layer $l-1$.

### Q2: Why is $\mathbf{b}^{[l]}$ shaped $(1, n^{[l]})$ and how does broadcasting handle batches?
**Answer**:
Each neuron in layer $l$ has exactly one scalar bias parameter, so there are $n^{[l]}$ biases. In batch matrix multiplication, adding shape $(1, n^{[l]})$ to $(M, n^{[l]})$ automatically replicates the bias vector across all $M$ samples via NumPy/TensorFlow broadcasting.

### Q3: How do you calculate the total number of trainable parameters in an MLP?
**Answer**:
For each layer $l$ from $1$ to $L$: $\text{Params}^{[l]} = (n^{[l-1]} \times n^{[l]}) + n^{[l]} = (n^{[l-1]} + 1) \times n^{[l]}$. Total parameters is the summation $\sum_{l=1}^L \text{Params}^{[l]}$.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Be vigilant on matrix dimension matching: $(M \times n_{l-1}) \times (n_{l-1} \times n_l) = (M \times n_l)$.
- Remember total trainable parameters formula: $(n_{in} + 1) \times n_{out}$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=H0_3SJh4Rqs)
- [Lecture Video](https://www.youtube.com/watch?v=H0_3SJh4Rqs)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
