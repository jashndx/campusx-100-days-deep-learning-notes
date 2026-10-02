# Lecture 058: Types of RNN | One-to-One, One-to-Many, Many-to-One, Many-to-Many

> **CampusX 100 Days of Deep Learning** | Video ID: `TkOBxzhIySg` | Duration: 26m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=TkOBxzhIySg) | **Transcript Status**: Available (en-US, 2825 words)

---

## 1. Executive Summary & Core Intuition
Recurrent Neural Networks are uniquely flexible because their input and output sequence dimensions can be decoupled.

Andrej Karpathy famously classified RNN applications into four canonical structural topologies based on input/output cardinalities:

1. **One-to-One:** Standard feedforward network (Image classification).

2. **One-to-Many:** Single input maps to variable sequence output (Image Captioning).

3. **Many-to-One:** Variable sequence input maps to single output (Sentiment Analysis, Video Action Classification).

4. **Many-to-Many (Synchronized / Same Length):** Ingests sequence, emits output at every time step (Part-of-Speech Tagging, Named Entity Recognition, Frame-by-frame video classification).

5. **Many-to-Many (Delayed / Asymmetric Length):** Ingests entire input sequence before emitting output sequence (Machine Translation, Speech Recognition, Chatbots) — the foundation of Encoder-Decoder (Seq2Seq) models!


## 2. Key Definitions & Formal Terminology
- **One-to-Many RNN**: An architecture that maps a single fixed-size input into a sequential stream of outputs across multiple time steps.
- **Many-to-Many (Synced)**: A sequence tagging topology where input length equals output length ($T_x = T_y$), with one output prediction emitted per input token.
- **Seq2Seq (Asymmetric Many-to-Many)**: An architecture composed of an Encoder reading $T_x$ inputs and a Decoder generating $T_y$ outputs ($T_x \neq T_y$).


## 3. Mathematical Formulations & Derivations
**Mathematical Mappings for RNN Topologies:**



1. **Many-to-One:**

   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t) \quad \forall t \in [1, T_x]$$

   $$\hat{\mathbf{y}} = g(\mathbf{h}_{T_x})$$



2. **Many-to-Many (Synchronized, $T_x = T_y$):**

   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t) \quad \forall t$$

   $$\hat{\mathbf{y}}_t = g(\mathbf{h}_t) \quad \forall t \in [1, T_x]$$

   $$\mathcal{L}_{total} = \sum_{t=1}^{T_x} \mathcal{L}(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$



3. **Many-to-Many (Delayed / Seq2Seq, $T_x \neq T_y$):**

   - Encoder: $\mathbf{h}_t^e = f_e(\mathbf{h}_{t-1}^e, \mathbf{x}_t) \implies \mathbf{c} = \mathbf{h}_{T_x}^e$ (Context Vector)

   - Decoder: $\mathbf{h}_t^d = f_d(\mathbf{h}_{t-1}^d, \mathbf{c}, \hat{\mathbf{y}}_{t-1})$

   - Predictions: $\hat{\mathbf{y}}_t = g(\mathbf{h}_t^d)$ for $t \in [1, T_y]$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Karpathy's Taxonomy Visual Diagram:**

```

1. One-to-One     2. One-to-Many       3. Many-to-One       4. Many-to-Many (Synced)   5. Many-to-Many (Seq2Seq)

     [y]              [y1] [y2] [y3]             [y]             [y1] [y2] [y3]             [y1] [y2] [y3]

      ^                ^    ^    ^                ^               ^    ^    ^                ^    ^    ^

      |                |    |    |                |               |    |    |                |    |    |

    [ANN]            [RNN]->[RNN]->[RNN]    [RNN]->[RNN]->[RNN] [RNN]->[RNN]->[RNN]     [Enc]->[Enc] -> [Dec]->[Dec]

      ^                ^                          ^    ^    ^     ^    ^    ^             ^    ^

      |                |                          |    |    |     |    |    |             |    |

     [x]              [x]                        [x1] [x2] [x3]  [x1] [x2] [x3]          [x1] [x2]

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# 1. Many-to-One (Sentiment Analysis)

many_to_one = models.Sequential([

    layers.SimpleRNN(64, return_sequences=False, input_shape=(None, 50)),

    layers.Dense(1, activation='sigmoid')

])



# 2. Many-to-Many Synchronized (POS Tagging: TimeDistributed)

many_to_many_synced = models.Sequential([

    layers.SimpleRNN(64, return_sequences=True, input_shape=(None, 50)),

    layers.TimeDistributed(layers.Dense(15, activation='softmax')) # 15 POS tags

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What real-world applications correspond to each of the 4 RNN topologies?
**Answer**:
1. **One-to-Many:** Image Captioning (Input: 1 image; Output: sentence of words). 2. **Many-to-One:** Sentiment Analysis, Spoken Intent Classification. 3. **Many-to-Many (Synced):** Part-of-Speech Tagging, Named Entity Recognition, Video frame segmentation. 4. **Many-to-Many (Asymmetric):** Machine Translation (English to French), Speech-to-Text transcription.

### Q2: Why can Machine Translation NOT be solved using a synchronized Many-to-Many RNN?
**Answer**:
Two reasons: (1) Different languages have different word counts (e.g., a 5-word English sentence may translate to 8 German words, so $T_x \neq T_y$). (2) Word order varies drastically across grammars (Subject-Verb-Object in English vs Subject-Object-Verb in German/Hindi); the network must read the *entire* sentence before emitting the first translated word.

### Q3: What is the role of `TimeDistributed` in Keras?
**Answer**:
`TimeDistributed(Dense(k))` applies the identical dense layer independently to every temporal slice of the sequence tensor $(M, T, H)$, producing output shape $(M, T, k)$ without flattening.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Know all 4 topologies and real-world examples of each.
- Machine translation requires delayed Many-to-Many (Seq2Seq), not synchronized.
- Use `TimeDistributed` for synchronized sequence tagging.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=TkOBxzhIySg)
- [Lecture Video](https://www.youtube.com/watch?v=TkOBxzhIySg)

---
