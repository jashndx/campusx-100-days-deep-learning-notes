# Lecture 061: LSTM (Long Short Term Memory) Part 1 | The What & Core Intuition

> **CampusX 100 Days of Deep Learning** | Video ID: `z7IPBg6MyrU` | Duration: 32m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=z7IPBg6MyrU) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997) is one of the most brilliant architectural innovations in the history of artificial intelligence.

LSTMs solved the vanishing gradient problem by redesigning the recurrent neuron from the ground up.

The Core Intuition: **The Conveyor Belt / Information Superhighway (Cell State $\mathbf{C}_t$)**.

In a vanilla RNN, the hidden state is violently overwritten at every step by non-linear matrix multiplication: $\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h} + \mathbf{W}\mathbf{x})$.

In an LSTM, the long-term memory (**Cell State $\mathbf{C}_t$**) runs straight down the entire chain with only minimal linear interactions!

Information can travel down this highway completely unmodified across hundreds of time steps.

To control what enters and leaves this highway, LSTMs introduce three specialized regulatory valves called **Gates** (Forget Gate, Input Gate, Output Gate), each powered by a Sigmoid function ($\sigma \in [0, 1]$).


## 2. Key Definitions & Formal Terminology
- **LSTM (Long Short-Term Memory)**: A specialized recurrent neural network architecture designed to learn long-term dependencies by regulating information flow through gated mechanisms and an additive cell state channel.
- **Cell State ($\mathbf{C}_t$)**: The internal memory conveyor belt of an LSTM that transports information across time with linear, additive updates, preventing gradient decay.
- **Hidden State ($\mathbf{h}_t$)**: The filtered working memory output of the LSTM cell at time $t$.
- **Gate**: A mechanism composed of a Sigmoid neural net layer and an element-wise multiplication that controls the proportion of information allowed to pass ($0 = \text{block completely}, 1 = \text{pass entirely}$).


## 3. Mathematical Formulations & Derivations
**The Fundamental Constant Error Carousel (CEC) Principle:**

Why does the Cell State eliminate vanishing gradients?

In an LSTM, the cell state update is **additive**, not multiplicative:

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \widetilde{\mathbf{C}}_t$$



When computing the derivative $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}}$:

$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$$



If the forget gate $\mathbf{f}_t \approx 1$ (remember):

$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = 1.0$$

The gradient flows across time steps by multiplying by $\mathbf{f} \approx 1.0$, completely avoiding exponential decay!

$$\prod_{k=j+1}^t \frac{\partial \mathbf{C}_k}{\partial \mathbf{C}_{k-1}} \approx 1.0^{t-j} = 1.0$$

The gradient highway remains open across hundreds of time steps!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**The Three Regulatory Gates:**

1. **Forget Gate ($\mathbf{f}_t$):** 'What old information should we discard from cell state?'

2. **Input Gate ($\mathbf{i}_t$ and $\widetilde{\mathbf{C}}_t$):** 'What new candidate information should we store in cell state?'

3. **Output Gate ($\mathbf{o}_t$):** 'What parts of the cell state should be emitted as current hidden state $\mathbf{h}_t$?'



```

           Cell State C_{t-1} ---------------------[ x ]------------------(+)--------------------> Cell State C_t

                                                     ^                     ^

                                                     |                     |

                                                [Forget Gate]         [Input Gate]

                                                     |                     |

           Hidden State h_{t-1} -\              [ Sigmoid ]     [ Sigmoid ] * [ Tanh ]

                                  +--> [Gates] --+-------------------------+---------> [ Output Gate: Sigmoid ] * Tanh(C_t) -> h_t

           Input x_t ------------/

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Instantiating an LSTM in Keras

# Input shape: (batch_size, timesteps, features)

model = models.Sequential([

    layers.LSTM(64, return_sequences=True, input_shape=(None, 100)),

    layers.LSTM(32, return_sequences=False),

    layers.Dense(1, activation='sigmoid')

])

model.summary()

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is the Cell State update in LSTMs additive rather than multiplicative?
**Answer**:
In vanilla RNNs, hidden state updates multiply by $\mathbf{W}_{hh}$ at every step, creating exponential decay during backprop. In LSTMs, the cell state update is additive ($\mathbf{C}_t = \mathbf{f} \mathbf{C}_{t-1} + \mathbf{i} \widetilde{\mathbf{C}}$). During backprop, the gradient distributes over additions, allowing error signals to flow back through time without shrinking.

### Q2: What does a gate output of 0 versus 1 mean physically?
**Answer**:
Gates use the Sigmoid function ($\sigma(z) \in [0, 1]$). An output of 0 means 'close the valve completely—let nothing pass'. An output of 1 means 'open the valve completely—let all information pass'. Intermediate values (e.g., 0.7) represent partial retention.

### Q3: Who invented the LSTM and in what year?
**Answer**:
Sepp Hochreiter and Jürgen Schmidhuber in their landmark 1997 paper 'Long Short-Term Memory', published in Neural Computation.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Cell state $\mathbf{C}_t$ is the constant error carousel (gradient highway).
- 3 gates: Forget ($\mathbf{f}_t$), Input ($\mathbf{i}_t$), Output ($\mathbf{o}_t$).
- Additive updates prevent the vanishing gradient problem.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=z7IPBg6MyrU)
- [Lecture Video](https://www.youtube.com/watch?v=z7IPBg6MyrU)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
