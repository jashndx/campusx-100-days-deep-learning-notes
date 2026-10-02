# Lecture 076: Why is Self Attention Called 'Self'? | Self vs Cross Attention

> **CampusX 100 Days of Deep Learning** | Video ID: `o4ZVA0TuDRg` | Duration: 26m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=o4ZVA0TuDRg) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Why is this mechanism explicitly termed **Self-Attention**?

This lecture clarifies the critical distinction between **Self-Attention** and **Cross-Attention** (traditional Bahdanau/Luong attention):

- **Cross-Attention (Inter-Attention):** Connects two *different* sequences. Queries come from Sequence B (Decoder target sentence), while Keys and Values come from Sequence A (Encoder source sentence).

- **Self-Attention (Intra-Attention):** Queries, Keys, and Values ALL come from the **exact same sequence**!

In Self-Attention, a sequence looks at *itself*: words inside the sentence attend to other words inside the *same* sentence to resolve pronouns, syntactic dependencies, and polysemy.


## 2. Key Definitions & Formal Terminology
- **Self-Attention (Intra-Attention)**: An attention mechanism where Queries, Keys, and Values are all derived from the same sequence: $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V} = \\mathbf{X}\\mathbf{W}_Q, \\mathbf{X}\\mathbf{W}_K, \\mathbf{X}\\mathbf{W}_V$.
- **Cross-Attention**: An attention mechanism where Queries originate from one sequence (e.g., decoder state), and Keys and Values originate from another sequence (e.g., encoder output).
- **Intra-Sentence Dependency**: Syntactic and semantic relationships that exist strictly within the boundaries of a single sentence (e.g., subject-verb agreement, antecedent resolution).


## 3. Mathematical Formulations & Derivations
**Mathematical Comparison of Origin Sources:**



1. **Self-Attention (Encoder or Decoder Self-Attention):**

   $$\mathbf{X} \in \mathbb{R}^{N \times d}$$

   $$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$

   *All three tensors originate from the single input $\mathbf{X}$!*



2. **Cross-Attention (Decoder-Encoder Attention):**

   Let $\mathbf{X}_{dec} \in \mathbb{R}^{M \times d}$ (Decoder sequence) and $\mathbf{H}_{enc} \in \mathbb{R}^{N \times d}$ (Encoder output):

   $$\mathbf{Q} = \mathbf{X}_{dec} \mathbf{W}_Q \quad (\text{From Decoder!})$$

   $$\mathbf{K} = \mathbf{H}_{enc} \mathbf{W}_K \quad (\text{From Encoder!})$$

   $$\mathbf{V} = \mathbf{H}_{enc} \mathbf{W}_V \quad (\text{From Encoder!})$$

   Queries probe across into the external encoder memory!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Comparison Matrix:**



| Feature | Self-Attention | Cross-Attention |

| :--- | :--- | :--- |

| **Input Source(s)** | Single sequence $\mathbf{X}$ | Two distinct sequences ($\mathbf{X}_{target}, \mathbf{X}_{source}$) |

| **Where it occurs in Transformer** | Encoder Layers & Decoder Layer 1 | Decoder Layer 2 (Encoder-Decoder block) |

| **Core Objective** | Contextualize words within same sequence | Align target words with source words |

| **Attention Matrix Shape** | $(N \times N)$ square | $(M \times N)$ rectangular ($M = T_{dec}, N = T_{enc}$) |


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



# In Keras MultiHeadAttention:

mha = layers.MultiHeadAttention(num_heads=8, key_dim=64)



# 1. Self-Attention: query=x, value=x, key=x (defaults to value if key not given)

self_att_out = mha(query=x, value=x, key=x)



# 2. Cross-Attention: query=decoder_state, value=encoder_state, key=encoder_state

cross_att_out = mha(query=decoder_tokens, value=encoder_memory, key=encoder_memory)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Where in the complete Transformer architecture does Self-Attention occur, and where does Cross-Attention occur?
**Answer**:
**Self-Attention** occurs in two places: (1) Inside every Encoder layer (unmasked bidirectional self-attention), and (2) Inside the first sublayer of every Decoder layer (masked causal self-attention). **Cross-Attention** occurs in the second sublayer of every Decoder layer, where queries from the decoder attend to keys and values from the encoder.

### Q2: Why is the attention matrix in Cross-Attention rectangular $(M \times N)$ rather than square $(N \times N)$?
**Answer**:
Because in Cross-Attention, the decoder sequence has length $M$ (target sentence length) and the encoder sequence has length $N$ (source sentence length). The dot product $\mathbf{Q}\mathbf{K}^T$ multiplies $(M \times d_k) \times (d_k \times N)$, producing an $(M \times N)$ matrix.

### Q3: Can Self-Attention be applied to non-text modalities?
**Answer**:
Yes! In Vision Transformers (ViT), image patches are treated as tokens that attend to other image patches within the *same* image (Self-Attention). In audio, temporal frames attend to other frames within the *same* audio waveform.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Self-Attention: Q, K, V all come from the same sequence $\mathbf{X}$.
- Cross-Attention: Q comes from Decoder; K, V come from Encoder.
- Cross-attention matrix shape is rectangular: $(T_{dec} \times T_{enc})$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=o4ZVA0TuDRg)
- [Lecture Video](https://www.youtube.com/watch?v=o4ZVA0TuDRg)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
