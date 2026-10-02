# Lecture 070: Bahdanau Attention Vs Luong Attention | Additive vs Multiplicative

> **CampusX 100 Days of Deep Learning** | Video ID: `0hZT4_fHfNQ` | Duration: 31m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=0hZT4_fHfNQ) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Following Bahdanau's breakthrough, Minh-Thang Luong et al. (2015) introduced key architectural refinements that simplified attention and improved computational efficiency.

This lecture contrasts the two classical attention paradigms:

1. **Bahdanau (Additive) Attention:** Uses a feedforward network (vector addition) to compute alignment scores. It calculates attention *before* updating the decoder hidden state, using $\mathbf{s}_{t-1}$.

2. **Luong (Multiplicative) Attention:** Replaces the expensive additive MLP with matrix dot products (General / Dot / Concat). It calculates attention *after* updating the decoder hidden state, using $\mathbf{s}_t$.

Multiplicative attention is mathematically faster and more memory efficient because it translates directly into highly optimized BLAS matrix multiplications on modern GPUs.


## 2. Key Definitions & Formal Terminology
- **Luong (Multiplicative) Attention**: An attention variant developed by Luong et al. (2015) that calculates alignment scores using dot products and utilizes the current decoder hidden state $\mathbf{s}_t$.
- **Global vs Local Attention**: Global attention attends to all source tokens; Local attention attends only to a small centered window $[p_t - D, p_t + D]$ of source tokens to save compute on long texts.
- **Dot-Product Alignment Score**: The simplest alignment score: $e_{ti} = \mathbf{s}_t^T \mathbf{h}_i$, which requires no trainable parameters and measures direct vector cosine similarity.


## 3. Mathematical Formulations & Derivations
**Comparison of Alignment Scoring Functions:**



Given decoder query $\mathbf{s}$ and encoder key $\mathbf{h}$:



1. **Bahdanau (Additive):**

   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s} + \mathbf{U}_a \mathbf{h})$$



2. **Luong (Dot):** *(Requires $\text{dim}(\mathbf{s}) = \text{dim}(\mathbf{h})$)*

   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{h}$$



3. **Luong (General):** *(Introduces learnable bilinear matrix $\mathbf{W}_a$)*

   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{W}_a \mathbf{h}$$



4. **Luong (Concat):**

   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}_a^T \tanh(\mathbf{W}_a [\mathbf{s} ; \mathbf{h}])$$



**Timing Difference:**

- Bahdanau: Uses $\mathbf{s}_{t-1}$ to compute $\mathbf{c}_t$, then computes $\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$.

- Luong: Computes $\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1})$ first, then uses $\mathbf{s}_t$ to compute $\mathbf{c}_t$, combining them into attentional vector:

  $$\widetilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t ; \mathbf{s}_t])$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Key Differences Summary Table:**



| Feature | Bahdanau Attention (2014) | Luong Attention (2015) |

| :--- | :--- | :--- |

| **Score Type** | Additive ($\mathbf{v}^T \tanh(\mathbf{W}\mathbf{s} + \mathbf{U}\mathbf{h})$) | Multiplicative ($\mathbf{s}^T \mathbf{W} \mathbf{h}$ or $\mathbf{s}^T \mathbf{h}$) |

| **Decoder State Used** | Previous state $\mathbf{s}_{t-1}$ | Current state $\mathbf{s}_t$ |

| **Computational Speed**| Slower (evaluates MLP) | Faster (BLAS matrix multiply) |

| **Scope** | Always Global | Global or Local |

| **Attentional Vector** | Emitted directly from RNN | Combined via $\widetilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t])$ |


## 5. Implementation Code Snippet
```python

import tensorflow as tf



# Luong Dot-Product Alignment Score

def luong_dot_score(decoder_state, encoder_states):

    # decoder_state: (batch, 1, dim)

    # encoder_states: (batch, seq_len, dim)

    # Transpose encoder states: (batch, dim, seq_len)

    # Matrix multiply computes dot product for all tokens simultaneously!

    score = tf.matmul(decoder_state, encoder_states, transpose_b=True)

    return score # shape: (batch, 1, seq_len)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Multiplicative (Dot) Attention faster than Additive Attention?
**Answer**:
Dot-product attention can be implemented as a single, highly optimized batch matrix multiplication (`tf.matmul` or `torch.bmm`), which fully saturates GPU matrix cores. Additive attention involves multiple linear projections, an element-wise addition, and a non-linear $\tanh$ activation, which are memory-bandwidth bound.

### Q2: What is the difference between Global and Local Attention in Luong et al.?
**Answer**:
**Global Attention:** Attends to *every* source token across the entire sentence, with computational complexity $\mathcal{O}(T_x)$ per decoding step. **Local Attention:** Predicts an aligned center position $p_t$ on the source sentence and computes attention strictly within a small window $[p_t - D, p_t + D]$, reducing computational complexity to a constant $\mathcal{O}(2D + 1)$.

### Q3: Which alignment score in Luong attention requires no trainable parameters?
**Answer**:
The **Dot** score: $\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{h}$. It computes direct cosine-like vector similarity, requiring zero additional weights, provided hidden dimensions match.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Bahdanau = Additive (uses $\mathbf{s}_{t-1}$); Luong = Multiplicative (uses $\mathbf{s}_t$).
- Multiplicative attention scales efficiently via GPU matrix multiplication.
- Luong General score: $\mathbf{s}^T \mathbf{W}_a \mathbf{h}$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=0hZT4_fHfNQ)
- [Lecture Video](https://www.youtube.com/watch?v=0hZT4_fHfNQ)

---
