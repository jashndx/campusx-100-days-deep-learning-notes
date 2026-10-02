# Lecture 064: Gated Recurrent Unit (GRU) | Modern Simplified Recurrent Cell

> **CampusX 100 Days of Deep Learning** | Video ID: `QQfZAoNGQmE` | Duration: 31m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=QQfZAoNGQmE) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
While LSTMs successfully solved vanishing gradients, their complexity is noticeable: 3 gates, a separate cell state, and $4\times$ the parameters of a vanilla RNN.

In 2014, Kyunghyun Cho et al. introduced the **Gated Recurrent Unit (GRU)** as a streamlined, faster alternative.

Key architectural simplifications of the GRU:

1. **Merges Cell State and Hidden State:** Eliminates the separate $\mathbf{C}_t$ channel; the hidden state $\mathbf{h}_t$ serves as both the memory carrier and working activation.

2. **Reduces Gates from 3 to 2:** Eliminates the separate Forget and Input gates, replacing them with a single coupled **Update Gate ($\mathbf{z}_t$)**. If the update gate decides to keep $80\%$ of old memory ($z_t = 0.8$), it automatically takes only $20\%$ of new candidate memory ($(1 - z_t) = 0.2$).

3. **Reset Gate ($\mathbf{r}_t$):** Controls how much of the previous state $\mathbf{h}_{t-1}$ to forget before computing the candidate state.

GRUs contain **25% fewer parameters** than LSTMs, train significantly faster, and achieve virtually identical empirical accuracy.


## 2. Key Definitions & Formal Terminology
- **Gated Recurrent Unit (GRU)**: A recurrent neural network variant that combines forget and input gates into a single update gate and merges cell state into hidden state.
- **Update Gate ($\mathbf{z}_t$)**: A gate that controls the linear interpolation between the previous hidden state and the new candidate hidden state: $\\mathbf{h}_t = (1 - \\mathbf{z}_t) \\odot \\mathbf{h}_{t-1} + \\mathbf{z}_t \\odot \\widetilde{\\mathbf{h}}_t$.
- **Reset Gate ($\mathbf{r}_t$)**: A gate that determines how much of the past hidden state should contribute to the candidate hidden state.
- **Coupled Gating**: The design choice where the retain factor ($1 - z$) and addition factor ($z$) sum strictly to $1.0$.


## 3. Mathematical Formulations & Derivations
**The Canonical GRU Equations (Cho et al., 2014):**



Given input $\mathbf{x}_t \in \mathbb{R}^d$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^h$:



1. **Update Gate:**

   $$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$



2. **Reset Gate:**

   $$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$



3. **Candidate Hidden State:**

   $$\widetilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \ \mathbf{x}_t] + \mathbf{b}_h)$$

   *(Notice: $\mathbf{r}_t$ directly masks the previous hidden state!)*



4. **Final Hidden State (Linear Interpolation):**

   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \widetilde{\mathbf{h}}_t$$



**Parameter Count Formula:**

GRUs have only 3 sets of weight matrices ($\mathbf{z}, \mathbf{r}, \mathbf{h}$):

$$\text{Params}_{GRU} = 3 \times \left[ h \cdot (d + h + 1) \right]$$

Exactly **$75\%$ the parameter count of an LSTM**!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**LSTM vs GRU Structural Comparison:**



| Dimension | LSTM | GRU |

| :--- | :--- | :--- |

| **Number of Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |

| **Memory Channels** | 2 ($\mathbf{C}_t$ Cell State, $\mathbf{h}_t$ Hidden State) | 1 ($\mathbf{h}_t$ Hidden State) |

| **Parameters** | $4 \times [h(d + h + 1)]$ | $3 \times [h(d + h + 1)]$ |

| **Compute Speed** | Slower (more matrix ops) | $\approx 25-30\%$ faster |

| **Memory Footprint** | Higher | Lower |

| **Performance** | Better on long, complex sequences | Equal or better on small/medium datasets |


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Building a GRU network in Keras

model = models.Sequential([

    layers.Embedding(input_dim=5000, output_dim=64, input_length=100),

    layers.GRU(64, return_sequences=False), # Fast 2-gate architecture

    layers.Dense(1, activation='sigmoid')

])

model.summary()

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: How does the GRU couple the forget and input gates?
**Answer**:
In an LSTM, the forget gate $\mathbf{f}_t$ and input gate $\mathbf{i}_t$ are computed independently, meaning the model could theoretically decide to remember 100% of old memory and add 100% of new memory. The GRU couples them into a single convex combination: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \widetilde{\mathbf{h}}_t$. Retention and addition strictly sum to 1.

### Q2: Calculate the parameter count of a `GRU(units=64)` layer with input dimension `10`.
**Answer**:
Using the formula: $\text{Params} = 3 \times [h(d + h + 1)] = 3 \times [64 \times (10 + 64 + 1)] = 3 \times [64 \times 75] = 3 \times 4,800 = \mathbf{14,400}$ parameters. (Compare to 19,200 for an LSTM of identical capacity).

### Q3: When should one choose GRU over LSTM?
**Answer**:
When computational resources or GPU memory are constrained, when working with smaller datasets where LSTMs might overfit due to higher parameter counts, or when training latency is a priority. For tasks requiring long-range tracking of syntactic state (like complex source code parsing), LSTMs often retain a slight edge.

## 7. Crucial Exam Takeaways & Common Pitfalls
- GRU has 2 gates (Reset $\mathbf{r}_t$, Update $\mathbf{z}_t$) and NO separate cell state.
- Parameters: $3 \times [h(d + h + 1)]$ (25% fewer parameters than LSTM).
- Trains faster and achieves comparable performance on most benchmarks.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=QQfZAoNGQmE)
- [Lecture Video](https://www.youtube.com/watch?v=QQfZAoNGQmE)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
