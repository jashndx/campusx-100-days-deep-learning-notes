# Lecture 078: Positional Encoding in Transformers | Sinusoidal Formulations

> **CampusX 100 Days of Deep Learning** | Video ID: `GeoQBNNqIbM` | Duration: 34m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=GeoQBNNqIbM) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Self-Attention is fundamentally **Permutation-Invariant**!

If you randomly scramble the word order of a sentence, the self-attention formula $\text{Softmax}(QK^T)V$ computes the exact same values, just row-permuted!

To a raw Transformer, "Dog bites man" and "Man bites dog" appear 100% mathematically identical!

In RNNs, word order was naturally baked into the sequential step-by-step recurrence.

Because Transformers process all tokens in parallel, order must be explicitly injected.

Vaswani et al. solved this by adding **Positional Encodings ($\mathbf{PE}$)** directly to the input word embeddings before the first layer:

$$\mathbf{X}_{input} = \mathbf{Embedding}(x) + \mathbf{PE}$$

Instead of learning static lookup tables, they designed an ingenious **Sinusoidal Positional Encoding** using sine and cosine waves of varying frequencies, allowing the network to extrapolate to arbitrary sequence lengths and easily learn relative positions!


## 2. Key Definitions & Formal Terminology
- **Positional Encoding ($\mathbf{PE}$)**: A tensor added to token embeddings that injects information about the absolute and relative position of tokens in a sequence.
- **Permutation Invariance**: A mathematical property where a function's output does not depend on the ordering of its input elements: $f(\pi(\mathbf{X})) = \pi(f(\mathbf{X}))$. Self-attention is permutation equivariant without positional encodings.
- **Sinusoidal Positional Encoding**: Deterministic positional encodings constructed using sine and cosine functions across geometric frequency progressions.
- **Relative Position Shift Property**: The mathematical property where $PE_{pos + k}$ can be expressed as a linear transformation of $PE_{pos}$, allowing the model to easily learn relative distances.


## 3. Mathematical Formulations & Derivations
**The Sinusoidal Positional Encoding Formulas (Vaswani et al., 2017):**



For token position $pos \in [0, N-1]$ and dimension index $i \in [0, \frac{d_{model}}{2} - 1]$:



$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{model}}} \right)$$

$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{model}}} \right)$$



Where wavelengths form a geometric progression from $2\pi$ to $10,000 \cdot 2\pi$.



**The Linear Transformation Property for Relative Positions:**

For any fixed offset $k$, there exists a linear transformation matrix $\mathbf{M}_k \in \mathbb{R}^{2 \times 2}$ such that:

$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$

Where $\omega_i = \frac{1}{10000^{2i/d}}$. 

This rotational property allows self-attention to attend to relative positions ($pos + k$) purely via linear operations!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Visualizing Positional Encoding Addition:**

```

Input Tokens:      "The"               "cat"               "sat"

                     |                   |                   |

Embeddings:      E("The")            E("cat")            E("sat")      (d_model = 512)

                     +                   +                   +

Pos Encodings:    PE(pos=0)           PE(pos=1)           PE(pos=2)    (Sinusoidal waves)

                     |                   |                   |

Combined Input:  X_0 (512-dim)       X_1 (512-dim)       X_2 (512-dim)

```


## 5. Implementation Code Snippet
```python

import numpy as np

import tensorflow as tf



def get_positional_encoding(seq_len, d_model):

    pe = np.zeros((seq_len, d_model))

    position = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)

    

    # Division term: 10000^(2i / d_model)

    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    

    pe[:, 0::2] = np.sin(position * div_term) # Even indices: sin

    pe[:, 1::2] = np.cos(position * div_term) # Odd indices: cos

    

    return tf.constant(pe, dtype=tf.float32)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Positional Encoding ADDED to word embeddings rather than CONCATENATED?
**Answer**:
Concatenating would increase the input dimension (e.g., from 512 to $512 + k$), increasing parameters across all subsequent weight matrices throughout the entire network. Vaswani et al. showed that because the embedding space has 512 dimensions, the semantic embedding information and positional wave signals occupy nearly orthogonal subspaces, allowing addition without corrupting semantic meaning.

### Q2: Why did the authors use sinusoidal functions instead of simple learned embeddings or normalized integers ($pos / N$)?
**Answer**:
Normalizing integers by $pos / N$ fails because the scale changes depending on sentence length $N$. Learned positional embeddings cannot extrapolate to sequence lengths longer than those seen during training. Sinusoidal encodings have a fixed periodic structure that allows the model to extrapolate to unseen sequence lengths while providing the relative position linear shift property.

### Q3: Why are different frequencies used across dimension $i$?
**Answer**:
Similar to the binary representation of integers where the least significant bit alternates rapidly ($0, 1, 0, 1$) and higher bits alternate slowly ($00, 11, 00$), the low dimensions of PE have high frequencies (fine-grained position changes) and high dimensions have low frequencies (coarse-grained position changes), giving every position a unique continuous fingerprint.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Self-attention without positional encoding is 100% permutation invariant.
- Even dimensions use $\sin$, odd dimensions use $\cos$.
- Linear rotational property allows model to attend to relative distances ($pos + k$).


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=GeoQBNNqIbM)
- [Lecture Video](https://www.youtube.com/watch?v=GeoQBNNqIbM)
- [Official Course Notes (Lectures 78-84)](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)

---
