# Lecture 069: Attention Mechanism | Bahdanau Additive Attention in Depth

> **CampusX 100 Days of Deep Learning** | Video ID: `rj5V6q6-XUM` | Duration: 45m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=rj5V6q6-XUM) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
How do you break the Seq2Seq information bottleneck?

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014/2015) published the revolutionary paper: "Neural Machine Translation by Jointly Learning to Align and Translate".

The Core Intuition: **Stop compressing the entire sentence into a single static vector!**

Instead of discarding intermediate encoder hidden states, the decoder keeps *all* encoder hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_{T_x})$.

When generating output word $t$, the decoder looks at its current state $\mathbf{s}_{t-1}$ and compares it with every encoder state $\mathbf{h}_i$ to compute **Attention Weights $\alpha_{ti}$** (a probability distribution summing to 1).

A dynamic, custom **Context Vector $\mathbf{c}_t$** is constructed as a weighted average: $\mathbf{c}_t = \sum \alpha_{ti} \mathbf{h}_i$.

When generating the word "cat", the network pays 95% attention to the encoder state for "chat" and 5% to other words!


## 2. Key Definitions & Formal Terminology
- **Attention Mechanism**: A mechanism that allows a decoder to dynamically focus on specific relevant parts of the input sequence when generating each output token.
- **Alignment Score ($e_{ti}$)**: A scalar quantifying how well the inputs around position $i$ match the output at position $t$.
- **Attention Weights ($\alpha_{ti}$)**: Softmax-normalized alignment scores representing the probability distribution over source tokens for decoding step $t$.
- **Dynamic Context Vector ($\mathbf{c}_t$)**: The weighted linear combination of all encoder hidden states: $\\mathbf{c}_t = \\sum_{i=1}^{T_x} \\alpha_{ti} \\mathbf{h}_i$.


## 3. Mathematical Formulations & Derivations
**The Complete Bahdanau (Additive) Attention Equations:**



Let encoder hidden states be $\mathbf{h}_1, \dots, \mathbf{h}_{T_x} \in \mathbb{R}^h$, and current decoder state be $\mathbf{s}_{t-1} \in \mathbb{R}^s$.



1. **Alignment Model (Additive / MLP Score):**

   $$e_{ti} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$

   Where $\mathbf{W}_a \in \mathbb{R}^{a \times s}, \mathbf{U}_a \in \mathbb{R}^{a \times h}, \mathbf{v}_a \in \mathbb{R}^{a \times 1}$ are learnable attention parameters.



2. **Softmax Normalization (Attention Weights):**

   $$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T_x} \exp(e_{tk})}, \quad \sum_{i=1}^{T_x} \alpha_{ti} = 1$$



3. **Dynamic Context Vector Computation:**

   $$\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{ti} \mathbf{h}_i$$



4. **Decoder State & Prediction Update:**

   $$\mathbf{s}_t = f_d([\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t])$$

   $$\hat{y}_t = \text{Softmax}(\mathbf{W}_o [\mathbf{s}_t, \mathbf{c}_t])$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Bahdanau Attention Dynamic Flowchart:**

```

Encoder States:     h_1        h_2        h_3        h_4

                     |          |          |          |

Alignment Scores:  e_t1       e_t2       e_t3       e_t4  <--- Compared with Decoder State s_{t-1}

                     |          |          |          |

Softmax:          a_t1       a_t2       a_t3       a_t4   (Sums to 1.0)

                     \          \          /          /

Dynamic Context:                  c_t = SUM(a_ti * h_i)

                                           |

                                           v

                              Decoder Step t ---> Emits y_t

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



class BahdanauAttention(layers.Layer):

    def __init__(self, units):

        super().__init__()

        self.W = layers.Dense(units)

        self.U = layers.Dense(units)

        self.V = layers.Dense(1)



    def call(self, query, values):

        # query: decoder hidden state s_{t-1} -> shape: (batch, 1, hidden_dim)

        # values: encoder hidden states h -> shape: (batch, seq_len, hidden_dim)

        

        # score = V^T * tanh(W*query + U*values)

        score = self.V(tf.nn.tanh(self.W(query) + self.U(values)))

        

        # attention_weights shape: (batch, seq_len, 1)

        attention_weights = tf.nn.softmax(score, axis=1)

        

        # context_vector shape: (batch, hidden_dim)

        context_vector = attention_weights * values

        context_vector = tf.reduce_sum(context_vector, axis=1)

        

        return context_vector, attention_weights

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: How does Bahdanau Attention eliminate the Seq2Seq Information Bottleneck?
**Answer**:
Instead of compressing all input tokens into one fixed vector $\mathbf{c} = \mathbf{h}_{T_x}$, the attention mechanism retains *all* intermediate encoder representations. At every decoding step, a bespoke context vector $\mathbf{c}_t$ is synthesized on-the-fly, allowing the decoder to look directly at any token in the source sentence regardless of sentence length.

### Q2: Why is Bahdanau Attention called 'Additive' Attention?
**Answer**:
Because inside the alignment scoring function, the decoder state and encoder state are projected and combined via vector addition: $\mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$. Contrast this with Luong's Multiplicative attention which uses dot products.

### Q3: How do Attention Weights provide model interpretability?
**Answer**:
Plotting the matrix of attention weights $\alpha_{ti}$ as a 2D heatmap produces an explicit word-alignment matrix showing exactly which source words the model focused on when emitting each translated target word (e.g., showing how English adjective-noun order maps to French noun-adjective order).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $\mathbf{c}_t = \sum \alpha_{ti} \mathbf{h}_i$.
- Attention weights sum to 1.0 via Softmax over source sequence $T_x$.
- Alignment heatmap provides direct visual explainability.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=rj5V6q6-XUM)
- [Lecture Video](https://www.youtube.com/watch?v=rj5V6q6-XUM)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
