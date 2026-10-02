# Lecture 073: Self-Attention in Transformers | Full Mathematical Derivation & Code

> **CampusX 100 Days of Deep Learning** | Video ID: `-tCKPl_8Xb8` | Duration: 41m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=-tCKPl_8Xb8) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
This lecture establishes the complete mathematical formulation of the Scaled Dot-Product Attention mechanism and derives its vectorized matrix implementation from scratch.

Tracing the full computational sequence for input matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$:

1. Project $\mathbf{X}$ into $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ via learned weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$.

2. Compute raw affinity scores matrix: $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$.

3. Scale by $\frac{1}{\sqrt{d_k}}$ to maintain unit variance and prevent gradient vanishing in Softmax.

4. Apply row-wise Softmax to compute the Attention Matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$.

5. Multiply attention weights by Value matrix: $\mathbf{Z} = \mathbf{A}\mathbf{V}$.

The entire multi-token, bidirectional contextualization is computed in **two matrix multiplications**!


## 2. Key Definitions & Formal Terminology
- **Scaled Dot-Product Attention**: The mathematical core of the Transformer: $\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V}$.
- **Projection Matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$)**: Learnable weight matrices of shape $(d_{model}, d_k)$ used to project input embeddings into query, key, and value subspaces.
- **Attention Matrix ($\mathbf{A}$)**: A square $N \times N$ stochastic matrix where entry $A_{i, j}$ represents the attention weight that token $i$ pays to token $j$ (each row sums to 1.0).


## 3. Mathematical Formulations & Derivations
**The Fundamental Transformer Equation:**



$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad (\mathbf{Q} \in \mathbb{R}^{N \times d_k})$$

$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \quad (\mathbf{K} \in \mathbb{R}^{N \times d_k})$$

$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \quad (\mathbf{V} \in \mathbb{R}^{N \times d_v})$$



$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$



**Dimensionality Dimensional Analysis:**

- Input sequence: $\mathbf{X} \in \mathbb{R}^{N \times d_{model}}$ ($N$ tokens, $d_{model}=512$)

- Weights: $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{model} \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_v}$

- Query $\times$ Key Transpose:

  $$\mathbf{Q}\mathbf{K}^T = (N \times d_k) \times (d_k \times N) = \mathbf{(N \times N)}$$

- Softmax over rows:

  $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$$

- Attention $\times$ Value:

  $$\mathbf{Z} = \mathbf{A}\mathbf{V} = (N \times N) \times (N \times d_v) = \mathbf{(N \times d_v)}$$

Output retains the exact same sequence length $N$ as the input!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Step-by-Step Computational Graph:**

```

Input X (N x d) ---> [ W_Q, W_K, W_V ] ---> Q (N x d_k), K (N x d_k), V (N x d_v)

                                                     |         |

                                                     v         v

                                              Matrix Mul: Q * K^T (N x N)

                                                           |

                                                           v

                                              Scale: / sqrt(d_k)

                                                           |

                                                           v

                                              Softmax (row-wise) ---> Attention Matrix A (N x N)

                                                                            |

                                                                            v

                                                                 Matrix Mul: A * V (N x d_v)

                                                                            |

                                                                            v

                                                               Output Contextualized Z (N x d_v)

```


## 5. Implementation Code Snippet
```python

import numpy as np



def scaled_dot_product_attention(Q, K, V, mask=None):

    d_k = Q.shape[-1]

    # 1. Matmul Q and K^T

    scores = np.matmul(Q, K.swapaxes(-2, -1))

    # 2. Scale by sqrt(d_k)

    scaled_scores = scores / np.sqrt(d_k)

    # 3. Optional Masking (for decoder causal attention)

    if mask is not None:

        scaled_scores += (mask * -1e9)

    # 4. Softmax over last axis

    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))

    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. Multiply by V

    output = np.matmul(attention_weights, V)

    return output, attention_weights

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Calculate the shape of the attention matrix $\mathbf{A}$ for a sequence of 50 words with embedding size 512.
**Answer**:
The attention matrix $\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)$ has shape $(N \times N) = \mathbf{50 \times 50}$. Every entry $A_{ij}$ indicates how much attention word $i$ pays to word $j$.

### Q2: Why must Softmax be applied across rows (`axis=-1`) rather than columns?
**Answer**:
Row $i$ of the matrix corresponds to the Query of token $i$ looking across all Keys $j \in \{1, \dots, N\}$. The attention distribution for token $i$ must form a valid probability distribution over all tokens $j$ ($\sum_{j=1}^N A_{ij} = 1.0$), which requires normalizing across rows.

### Q3: How does Self-Attention handle sentences of different lengths in a batch?
**Answer**:
Padding tokens are masked out before Softmax. A large negative number ($-10^9$ or $-\infty$) is added to the scores of padding positions. When exponentiated ($e^{-\infty} = 0$), the attention weights for pad tokens become exactly zero, ensuring real tokens ignore padding.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.
- Attention matrix shape is $(N \times N)$, independent of embedding dimension $d$.
- Output shape $(N \times d_v)$ matches input sequence length.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=-tCKPl_8Xb8)
- [Lecture Video](https://www.youtube.com/watch?v=-tCKPl_8Xb8)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
