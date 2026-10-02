# Lecture 066: Bidirectional RNN | BiLSTM & BiGRU Architectures

> **CampusX 100 Days of Deep Learning** | Video ID: `k2NSm3MNdYg` | Duration: 29m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=k2NSm3MNdYg) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
In standard unidirectional RNNs, the hidden state $\mathbf{h}_t$ only has access to the *past* context ($x_1, \dots, x_t$). It knows nothing about the future!

However, in many real-world NLP tasks (e.g., Named Entity Recognition, translation, sentiment analysis), understanding a word depends fundamentally on the words that come *after* it!

Consider the polysemous sentence:

- "The **bank** of the river was muddy." (Financial bank or river bank? You only know after reading "river"!)

- "He had to **bear** the heavy load." vs "The grizzly **bear** attacked."

**Bidirectional RNNs (BiRNN / BiLSTM / BiGRU)** (Schuster & Paliwal, 1997) solve this by training two independent recurrent networks in parallel:

1. A **Forward RNN** that reads the sequence from left-to-right ($t = 1 \to T$).

2. A **Backward RNN** that reads the sequence from right-to-left ($t = T \to 1$).

At each time step $t$, the forward hidden state $\overrightarrow{\mathbf{h}}_t$ and backward hidden state $\overleftarrow{\mathbf{h}}_t$ are concatenated, providing complete past and future contextual awareness.


## 2. Key Definitions & Formal Terminology
- **Bidirectional RNN (BiRNN)**: A neural network architecture that combines two independent recurrent layers running in opposite directions, allowing predictions to depend on both preceding and succeeding sequence context.
- **Forward Hidden State ($\overrightarrow{\mathbf{h}}_t$)**: The representation encoding sequence context from the past ($1$ to $t$).
- **Backward Hidden State ($\overleftarrow{\mathbf{h}}_t$)**: The representation encoding sequence context from the future ($T$ down to $t$).
- **Context Concatenation**: Combining both states into a single unified representation: $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2h}$.


## 3. Mathematical Formulations & Derivations
**Mathematical Formulation of Bidirectional RNN:**



Given sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$:



1. **Forward Hidden State Pass ($t = 1, \dots, T$):**

   $$\overrightarrow{\mathbf{h}}_t = \tanh\left( \mathbf{W}_{x\vec{h}} \mathbf{x}_t + \mathbf{W}_{\vec{h}\vec{h}} \overrightarrow{\mathbf{h}}_{t-1} + \mathbf{b}_{\vec{h}} \right)$$



2. **Backward Hidden State Pass ($t = T, \dots, 1$):**

   $$\overleftarrow{\mathbf{h}}_t = \tanh\left( \mathbf{W}_{x\overleftarrow{h}} \mathbf{x}_t + \mathbf{W}_{\overleftarrow{h}\overleftarrow{h}} \overleftarrow{\mathbf{h}}_{t+1} + \mathbf{b}_{\overleftarrow{h}} \right)$$



3. **Combined Output Representation:**

   $$\mathbf{h}_t = \left[ \overrightarrow{\mathbf{h}}_t \, ; \, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{2h}$$

   $$\hat{\mathbf{y}}_t = g\left( \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y \right)$$



**Parameter Count Formula:**

Because a Bidirectional layer instantiates two completely separate, independent recurrent cells:

$$\text{Params}_{\text{BiLSTM}} = 2 \times \text{Params}_{\text{LSTM}} = 8 \times [h(d + h + 1)]$$

Exactly double the parameters of a unidirectional cell!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Bidirectional Flow Architecture:**

```

Forward State:   --> [-> h_1] -------> [-> h_2] -------> [-> h_3] -->

                          |                 |                 |

Combined:            [ Concat ]        [ Concat ]        [ Concat ] ---> Output y_t

                          |                 |                 |

Backward State: <-- [<- h_1] <------- [<- h_2] <------- [<- h_3] <--

                          ^                 ^                 ^

Inputs:                  x_1               x_2               x_3

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Bidirectional LSTM in Keras

model = models.Sequential([

    layers.Embedding(input_dim=10000, output_dim=64, input_length=150),

    # Bidirectional wrapper doubles output units: 64 forward + 64 backward = 128!

    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),

    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),

    layers.Dense(1, activation='sigmoid')

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: When can Bidirectional RNNs NOT be used?
**Answer**:
In **Real-time Online Streaming tasks** and **Autoregressive Generation** (e.g., real-time speech transcription, live stock trading, next-token text prediction). In these tasks, future tokens have not occurred yet, making backward propagation physically impossible. BiRNNs require the *complete* sequence to be available upfront.

### Q2: Why does a `Bidirectional(LSTM(64))` output a 128-dimensional vector?
**Answer**:
By default, Keras concatenates the 64-dimensional forward hidden state $\overrightarrow{\mathbf{h}}$ with the 64-dimensional backward hidden state $\overleftarrow{\mathbf{h}}$, yielding $64 + 64 = 128$ dimensions (`merge_mode='concat'`).

### Q3: What are the alternative `merge_mode` options in Keras Bidirectional layers?
**Answer**:
`'concat'` (default, doubles dimension), `'sum'` (adds states, keeps 64 dims), `'mul'` (multiplies states), and `'ave'` (averages states).

## 7. Crucial Exam Takeaways & Common Pitfalls
- BiRNNs combine forward and backward context ($\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$).
- Doubles the parameter count of a unidirectional layer.
- Cannot be used for real-time online streaming or autoregressive generation.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=k2NSm3MNdYg)
- [Lecture Video](https://www.youtube.com/watch?v=k2NSm3MNdYg)

---
