# Lecture 077: Multi-Head Attention in Transformers | Multi-Head vs Self Attention

> **CampusX 100 Days of Deep Learning** | Video ID: `bX2QwpjsmuA` | Duration: 36m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=bX2QwpjsmuA) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
While single-head self-attention is powerful, it has a serious limitation: average pooling.

If a word attends to multiple relationships simultaneously (e.g., in "The bank was robbed on Tuesday", the word "bank" needs to attend to "robbed" for semantic meaning, and "was" for syntactic tense), a single attention head is forced to average these different relationships together, diluting nuanced linguistic signals.

**Multi-Head Attention (MHA)** solves this by running $h$ independent self-attention mechanisms (heads) in parallel!

Each head projects $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into a lower-dimensional subspace ($d_k = d_{model} / h$):

- Head 1 can focus on grammatical subject-verb agreement.

- Head 2 can focus on coreference resolution (pronouns).

- Head 3 can focus on direct object relationships.

The outputs of all $h$ heads are concatenated and multiplied by a final projection matrix $\mathbf{W}_O$, synthesizing multi-faceted contextual representations at **zero additional computational cost** compared to a full-rank single head!


## 2. Key Definitions & Formal Terminology
- **Multi-Head Attention (MHA)**: An attention module that linearly projects queries, keys, and values $h$ times with different learned projections, computes scaled dot-product attention in parallel, concatenates the results, and projects again.
- **Head ($h$)**: One of the parallel attention subspaces (standard: $h=8$ heads in base Transformer, $h=16$ in large models).
- **Head Dimension ($d_k$)**: The dimensionality of each individual head: $d_k = d_v = d_{model} / h$ (for $d_{model}=512$ and $h=8 \implies d_k = 64$).
- **Output Projection Matrix ($\mathbf{W}_O$)**: A learnable matrix of shape $(h \cdot d_v, d_{model})$ that linearly combines the concatenated multi-head outputs back into the model dimension.


## 3. Mathematical Formulations & Derivations
**Complete Multi-Head Attention Formulations (Vaswani et al., 2017):**



$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$



Where each individual head is:

$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \ \mathbf{K}\mathbf{W}_i^K, \ \mathbf{V}\mathbf{W}_i^V)$$



**Projection Matrix Dimensions:**

- $\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}$

- $\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$

- $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$

- $\mathbf{W}_O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}}$



**Parameter Calculation:**

For $h$ heads, each with $d_k = d_{model} / h$:

- Total weights for $Q, K, V$: $3 \times (h \times d_{model} \times d_k) = 3 \times (d_{model} \times d_{model}) = \mathbf{3 d_{model}^2}$.

- Output projection $\mathbf{W}_O$: $(h \cdot d_v) \times d_{model} = d_{model} \times d_{model} = \mathbf{d_{model}^2}$.

- **Total Parameters:** $4 d_{model}^2$ (+ biases).

*For $d_{model} = 512$:*

$$\text{Params}_{MHA} = 4 \times 512^2 = 4 \times 262,144 = \mathbf{1,048,576 \text{ Parameters}} \approx \mathbf{1.05 \text{ Million}}.$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Multi-Head Attention Parallel Execution Flowchart:**

```

Input (N x d_model)

       |

  +----+----+----+----+----+----+----+----+ (Split into h=8 parallel heads)

  |    |    |    |    |    |    |    |

Head1 Head2 Head3 Head4 Head5 Head6 Head7 Head8  (Each computes Attention in d_k=64)

  |    |    |    |    |    |    |    |

  +----+----+----+----+----+----+----+----+

       |

  [ Concatenate: (N x 8*64) = (N x 512) ]

       |

  [ Linear Projection W_O: (512 x 512) ]

       |

  Output (N x d_model)

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



# MultiHeadAttention in Keras

mha_layer = layers.MultiHeadAttention(

    num_heads=8,     # h = 8

    key_dim=64       # d_k = 64 (Total model dim = 8 * 64 = 512)

)



# Input tensor shape: (batch, seq_len, 512)

x = tf.random.normal((32, 50, 512))

out = mha_layer(query=x, value=x, key=x)

print("MHA Output shape:", out.shape) # (32, 50, 512)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does Multi-Head Attention NOT increase computational complexity compared to a full-rank single-head attention?
**Answer**:
Because the representation dimension is divided across the heads: $d_k = d_{model} / h$. Computing $h$ dot products of size $d_k$ requires $h \times (N^2 \cdot d_k) = N^2 \cdot (h \cdot d_k) = N^2 \cdot d_{model}$ operations—the exact same computational FLOP count as a single head operating on the full dimension $d_{model}$!

### Q2: What is the primary representational advantage of Multi-Head Attention over Single-Head Attention?
**Answer**:
Single-head attention forces the network to average attention over multiple conflicting linguistic relationships. Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions simultaneously (e.g., syntactic structure in head 1, coreference in head 2, semantic theme in head 3).

### Q3: Calculate the total trainable parameters in a Multi-Head Attention layer with $d_{model}=768$ and $h=12$ (BERT-Base specifications).
**Answer**:
Using formula $\text{Params} \approx 4 \times d_{model}^2$: $4 \times (768)^2 = 4 \times 589,824 = \mathbf{2,359,296}$ weights (plus $4 \times 768 = 3,072$ biases).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $d_k = d_{model} / h$.
- MHA parameter count is $4 d_{model}^2$.
- FLOP complexity of $h$ heads at dimension $d/h$ is identical to 1 head at dimension $d$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=bX2QwpjsmuA)
- [Lecture Video](https://www.youtube.com/watch?v=bX2QwpjsmuA)
- [Colab Notebook](https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ)

---
