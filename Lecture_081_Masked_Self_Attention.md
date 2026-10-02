# Lecture 081: Masked Self-Attention | Look-Ahead Causal Masking in Decoder

> **CampusX 100 Days of Deep Learning** | Video ID: `m6onaKFzF94` | Duration: 33m 50s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=m6onaKFzF94) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
In the Encoder, self-attention is bi-directional: every token attends to all past and future tokens (e.g., word 2 looks at word 5).

However, during text generation in the Decoder, the task is **Autoregressive**: when predicting word $t$, the future words $t+1, t+2, \dots$ have not been generated yet!

During training, we feed the entire target sentence into the decoder simultaneously to leverage GPU parallelism.

If the decoder used standard self-attention, token $t$ would simply 'cheat' and look ahead at token $t+1$ (data leakage), learning zero predictive ability!

To prevent future leakage while preserving parallel training, we use **Masked Self-Attention (Causal Masking)**.

A lower-triangular matrix masks out future positions by setting their attention logits to $-\infty$.

When passed through Softmax, $e^{-\infty} = 0$, guaranteeing that token $t$ can attend *only* to tokens $\le t$!


## 2. Key Definitions & Formal Terminology
- **Masked Self-Attention (Causal Attention)**: A modified self-attention mechanism that enforces causality by masking future positions, ensuring predictions for position $t$ depend only on known outputs at positions prior to $t$.
- **Look-Ahead Mask (Causal Mask)**: An upper-triangular matrix of $-\\infty$ (or zeros) used to zero out attention weights for all positions $j > i$.
- **Information Leakage (Cheating)**: A failure mode where an autoregressive model accesses future ground truth target tokens during training, rendering it incapable of generating text at inference.


## 3. Mathematical Formulations & Derivations
**Mathematical Formulation of Causal Masking:**



Let raw scaled dot-product attention scores be $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$.

Define the Causal Mask $\mathbf{M} \in \mathbb{R}^{N \times N}$:



$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \quad (\text{Past and Present: Allowed}) \\ -\infty & \text{if } j > i \quad (\text{Future: Masked!}) \end{cases}$$



**Masked Attention Calculation:**

$$\mathbf{A} = \text{Softmax}(\mathbf{S} + \mathbf{M})$$



For an element in the future ($j > i$):

$$A_{i, j} = \frac{\exp(S_{i, j} + (-\infty))}{\sum_{k} \exp(S_{i, k} + M_{i, k})} = \frac{0}{\sum_{k \le i} \exp(S_{i, k})} = \mathbf{0.0}$$

Attention weight to all future tokens is strictly zero!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Causal Mask Matrix Structure ($N=4$ tokens):**

```

Tokens:      [w1]   [w2]   [w3]   [w4]

Row 1 (w1): [  0    -inf   -inf   -inf ]  ---> Attends ONLY to w1

Row 2 (w2): [  0      0    -inf   -inf ]  ---> Attends to w1, w2

Row 3 (w3): [  0      0      0    -inf ]  ---> Attends to w1, w2, w3

Row 4 (w4): [  0      0      0      0  ]  ---> Attends to w1, w2, w3, w4



After Softmax:

Row 1:      [ 1.0    0.0    0.0    0.0 ]

Row 2:      [ 0.4    0.6    0.0    0.0 ]

Row 3:      [ 0.2    0.3    0.5    0.0 ]

Row 4:      [ 0.1    0.2    0.3    0.4 ]

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf



def create_causal_mask(seq_len):

    # Upper triangular matrix with ones above main diagonal

    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    # Convert ones to -1e9 (-infinity)

    return mask * -1e9



# Example for seq_len = 4

mask = create_causal_mask(4)

print("Causal Mask:\n", mask.numpy())

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is $-\infty$ used in the mask rather than $0$ before Softmax?
**Answer**:
Softmax exponentiates its inputs: $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$. If you added $0$, $e^0 = 1$, which would give future tokens positive probability! Adding $-\infty$ ensures $e^{-\infty} = 0$, guaranteeing the attention weight for future tokens is absolute zero.

### Q2: Why is Causal Masking unnecessary in the Encoder?
**Answer**:
The Encoder's job is full sentence comprehension (representation learning). It has access to the entire input sentence simultaneously, and words need bidirectional context (future and past words) to disambiguate meaning. Causal masking is required *only* in autoregressive Decoders.

### Q3: How does Causal Masking enable parallel training for generative models?
**Answer**:
Without masking, an autoregressive model would have to be trained sequentially one token at a time (like an RNN). With causal masking, all $T$ target tokens can be fed into the GPU simultaneously; the mask enforces that step $t$ cannot see step $t+1$, allowing all $T$ predictions to be computed and loss evaluated in a single parallel pass!

## 7. Crucial Exam Takeaways & Common Pitfalls
- Causal masking adds $-\infty$ to upper triangle ($j > i$).
- Ensures $e^{-\infty} = 0$ in Softmax; future attention weights are zero.
- Enables parallel training of autoregressive models.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=m6onaKFzF94)
- [Lecture Video](https://www.youtube.com/watch?v=m6onaKFzF94)
- [Official Course Notes](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)

---
