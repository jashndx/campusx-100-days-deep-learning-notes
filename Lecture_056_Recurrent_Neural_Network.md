# Lecture 056: Recurrent Neural Network | Forward Propagation & Architecture

> **CampusX 100 Days of Deep Learning** | Video ID: `BjWqCcbusMM` | Duration: 35m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=BjWqCcbusMM) | **Transcript Status**: Available (hi, 3228 words)

---

## 1. Executive Summary & Core Intuition
This lecture establishes the complete mathematical engine of the Vanilla Recurrent Neural Network (Elman RNN).

An RNN processes a sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ sequentially.

At each time step $t$:

1. It receives the current input vector $\mathbf{x}_t$ and the previous hidden state vector $\mathbf{h}_{t-1}$.

2. It linearly combines them using two weight matrices: input-to-hidden $\mathbf{W}_{xh}$ and hidden-to-hidden $\mathbf{W}_{hh}$.

3. It applies a non-linear activation (almost universally **Tanh**) to produce the updated hidden memory state $\mathbf{h}_t$.

4. If an output is required at step $t$, $\mathbf{h}_t$ is projected via hidden-to-output matrix $\mathbf{W}_{hy}$ into output prediction $\hat{\mathbf{y}}_t$.

Unrolling the recurrent loop across time transforms the RNN into an equivalent deep feedforward network where depth equals the number of time steps $T$!


## 2. Key Definitions & Formal Terminology
- **Vanilla / Elman RNN**: The standard recurrent architecture introduced by Jeffrey Elman in 1990 where current hidden state is a function of current input and previous hidden state.
- **Unrolling / Unfolding in Time**: Representing a recurrent network as a sequence of identical interconnected feedforward layers, one for each time step in the input sequence.
- **Hidden-to-Hidden Weight Matrix ($\mathbf{W}_{hh}$)**: The square transition matrix that maps the previous memory state $\mathbf{h}_{t-1}$ to the current state $\mathbf{h}_t$.
- **Input-to-Hidden Weight Matrix ($\mathbf{W}_{xh}$)**: The matrix that projects incoming feature vector $\mathbf{x}_t$ into the hidden state space.


## 3. Mathematical Formulations & Derivations
**The Fundamental Vanilla RNN Equations:**



At time step $t \in \{1, 2, \dots, T\}$, with initial state $\mathbf{h}_0 = \mathbf{0}$:



1. **Pre-activation:**

   $$\mathbf{a}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$$



2. **Hidden State Activation (Tanh):**

   $$\mathbf{h}_t = \tanh(\mathbf{a}_t) = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$



3. **Output Prediction (Softmax / Linear):**

   $$\hat{\mathbf{y}}_t = \text{Softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$



**Dimensionality Analysis:**

Let input feature dimension be $d$, hidden state dimension be $h$, and output dimension be $k$:

- $\mathbf{x}_t \in \mathbb{R}^{d \times 1}$

- $\mathbf{h}_t \in \mathbb{R}^{h \times 1}$

- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$

- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$

- $\mathbf{b}_h \in \mathbb{R}^{h \times 1}$

- $\mathbf{W}_{hy} \in \mathbb{R}^{k \times h}$

- $\mathbf{b}_y \in \mathbb{R}^{k \times 1}$



**Total Trainable Parameters:**

$$\text{Params} = (h \times d) + (h \times h) + h + (k \times h) + k$$

Notice that parameter count is completely independent of sequence length $T$!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Unrolling an RNN in Time ($T=3$ steps):**

```

     y_1                   y_2                   y_3

      ^                     ^                     ^

    [W_hy]                [W_hy]                [W_hy]

      |                     |                     |

h_0 ->[ Cell ]-- W_hh ---> [ Cell ]-- W_hh ---> [ Cell ] ---> h_3

      ^                     ^                     ^

    [W_xh]                [W_xh]                [W_xh]

      |                     |                     |

     x_1                   x_2                   x_3

```


## 5. Implementation Code Snippet
```python

import numpy as np



# Vanilla RNN Forward Pass from scratch in NumPy

class VanillaRNN:

    def __init__(self, d_in, d_hid, d_out):

        self.Wxh = np.random.randn(d_hid, d_in) * 0.01

        self.Whh = np.random.randn(d_hid, d_hid) * 0.01

        self.Why = np.random.randn(d_out, d_hid) * 0.01

        self.bh = np.zeros((d_hid, 1))

        self.by = np.zeros((d_out, 1))



    def forward(self, inputs):

        # inputs is a list of x_t vectors

        h = np.zeros((self.Whh.shape[0], 1))

        outputs = []

        for x in inputs:

            # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

            h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wxh, x) + self.bh)

            y = np.dot(self.Why, h) + self.by

            outputs.append(y)

        return outputs, h

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Tanh used as the activation function for hidden states in RNNs rather than ReLU?
**Answer**:
In an unrolled RNN of length $T$, the hidden state undergoes repeated matrix multiplications by $\mathbf{W}_{hh}$ at every single step ($h_T \approx \mathbf{W}_{hh}^T x_1$). If ReLU is used, unbounded activations ($z > 0$) can cause values to compound exponentially, leading to catastrophic numerical overflow ($+\infty$). Tanh squashes values strictly into $(-1, 1)$, keeping hidden representations bounded.

### Q2: How many trainable parameters are in a `SimpleRNN(units=100)` layer with input dimension `20`?
**Answer**:
Using the formula: $\text{Params} = (\text{units} \times \text{input\_dim}) + (\text{units} \times \text{units}) + \text{units} = (100 \times 20) + (100 \times 100) + 100 = 2,000 + 10,000 + 100 = \mathbf{12,100}$ parameters.

### Q3: What does the parameter `return_sequences=True` versus `return_sequences=False` control in Keras RNNs?
**Answer**:
`return_sequences=False` (default) outputs only the final hidden state vector $\mathbf{h}_T$ (shape $(M, \text{units})$), used for Many-to-One tasks like sentiment analysis. `return_sequences=True` outputs the full sequence of hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_T)$ (shape $(M, T, \text{units})$), required when stacking recurrent layers or for Many-to-Many sequence tagging.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Hidden state recurrence: $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$.
- Parameters: $h \cdot d + h^2 + h$ (independent of sequence length $T$).
- `return_sequences=True` outputs all timesteps $(M, T, h)$; `False` outputs only final step $(M, h)$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=BjWqCcbusMM)
- [Lecture Video](https://www.youtube.com/watch?v=BjWqCcbusMM)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
