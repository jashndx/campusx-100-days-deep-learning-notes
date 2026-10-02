# Lecture 065: Deep RNNs | Stacked RNNs, Stacked LSTMs & Stacked GRUs

> **CampusX 100 Days of Deep Learning** | Video ID: `mlDkTrlLaio` | Duration: 27m 50s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=mlDkTrlLaio) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Just as deep feedforward networks learn hierarchical representations across depth, recurrent networks can be stacked vertically to form **Deep (Stacked) RNNs**.

In a single-layer RNN, depth exists only horizontally across time.

By stacking recurrent layers:

- Layer 1 processes raw input vectors $\mathbf{x}_t$ and extracts low-level sequential patterns (e.g., character combinations, phonetic transitions).

- Layer 2 treats the hidden state sequence of Layer 1 as its input, learning higher-level temporal abstractions (e.g., word syntax, grammatical phrases).

- Layer 3 learns long-term semantic structures (e.g., discourse, narrative topic).

Critical implementation rule in Keras: **Every intermediate recurrent layer MUST have `return_sequences=True`** so it outputs a 3D temporal tensor `(batch, timesteps, units)` to feed the next recurrent layer!


## 2. Key Definitions & Formal Terminology
- **Deep / Stacked RNN**: A recurrent network architecture featuring multiple recurrent layers stacked on top of each other, where the hidden state sequence of layer $l-1$ serves as the input sequence to layer $l$.
- **Temporal Abstraction Hierarchy**: The property where lower layers in a stacked RNN operate on rapid, high-frequency temporal changes while deeper layers capture slow, abstract, long-range semantic patterns.
- **`return_sequences=True`**: The Keras parameter that dictates whether a recurrent layer outputs its full sequence of hidden states across all time steps or only the final hidden state vector.


## 3. Mathematical Formulations & Derivations
**Mathematical Formulation of Stacked Recurrent Layers:**



Let $\mathbf{h}_t^{[l]}$ denote the hidden state of layer $l$ at time step $t$:



For layer $l=1$:

$$\mathbf{h}_t^{[1]} = \text{RNN}\left(\mathbf{h}_{t-1}^{[1]}, \ \mathbf{x}_t\right)$$



For intermediate layers $l \in \{2, \dots, L\}$:

$$\mathbf{h}_t^{[l]} = \text{RNN}\left(\mathbf{h}_{t-1}^{[l]}, \ \mathbf{h}_t^{[l-1]}\right)$$



Notice that layer $l$ receives:

- Recurrent input from the past of its own layer: $\mathbf{h}_{t-1}^{[l]}$

- Feedforward input from the present of the previous layer: $\mathbf{h}_t^{[l-1]}$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Stacked 3-Layer LSTM Diagram:**

```

Output:                           y_t

                                   ^

Layer 3 (LSTM):   h_0^[3] -> [ Cell ] ------> [ Cell ] ------> h_T^[3]

                                   ^                ^

Layer 2 (LSTM):   h_0^[2] -> [ Cell ] ------> [ Cell ] ------> h_T^[2]

                                   ^                ^

Layer 1 (LSTM):   h_0^[1] -> [ Cell ] ------> [ Cell ] ------> h_T^[1]

                                   ^                ^

Inputs:                           x_1              x_T

```

Layer 1 and Layer 2 must set `return_sequences=True`. Layer 3 sets `return_sequences=False` (for Many-to-One).


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Deep Stacked LSTM Architecture in Keras

model = models.Sequential([

    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),

    # Layer 1: MUST return sequences!

    layers.LSTM(64, return_sequences=True),

    layers.Dropout(0.3),

    # Layer 2: MUST return sequences!

    layers.LSTM(64, return_sequences=True),

    layers.Dropout(0.3),

    # Layer 3: Final recurrent layer (Many-to-One)

    layers.LSTM(32, return_sequences=False),

    layers.Dense(1, activation='sigmoid')

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What error occurs in Keras if you stack two LSTM layers without setting `return_sequences=True` on the first?
**Answer**:
A fatal tensor shape mismatch error. A standard LSTM outputs a 2D tensor of shape `(batch_size, units)` representing only the final time step $\mathbf{h}_T$. The second LSTM layer requires a 3D sequential input tensor of shape `(batch_size, timesteps, features)`. Setting `return_sequences=True` ensures the first layer outputs `(batch_size, timesteps, units)`.

### Q2: Why do stacked RNNs rarely exceed 3 to 4 layers in depth?
**Answer**:
Training stacked RNNs compounds vanishing and exploding gradients in *two* directions simultaneously: horizontally across time $T$, and vertically across layers $L$. Without residual connections, stacking beyond 3-4 recurrent layers degrades training stability.

### Q3: How did Google's Neural Machine Translation (GNMT) system train an 8-layer stacked LSTM?
**Answer**:
Wu et al. (2016) introduced **Residual Connections between recurrent layers**: the input to layer $l$ was added directly to its output ($\mathbf{h}_t^{[l-1]} + \mathbf{h}_t^{[l]}$), enabling error gradients to bypass layers vertically without vanishing.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Always set `return_sequences=True` on all intermediate recurrent layers.
- Only the final recurrent layer sets `return_sequences=False` (for classification).
- Stacked RNNs create hierarchical feature representations across depth.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=mlDkTrlLaio)
- [Lecture Video](https://www.youtube.com/watch?v=mlDkTrlLaio)
- [Colab Notebook](https://colab.research.google.com/drive/1c4eN4cPxajCpFG6yr1mAUi3sV1JaGLDY?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
