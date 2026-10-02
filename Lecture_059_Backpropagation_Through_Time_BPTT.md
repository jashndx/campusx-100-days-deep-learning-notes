# Lecture 059: Backpropagation Through Time (BPTT) | Mathematical Derivation

> **CampusX 100 Days of Deep Learning** | Video ID: `OvCz1acvt-k` | Duration: 42m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=OvCz1acvt-k) | **Transcript Status**: Available (en-US, 4128 words)

---

## 1. Executive Summary & Core Intuition
How do you train an unrolled RNN whose parameters are shared across dozens of time steps?

The answer is **Backpropagation Through Time (BPTT)**.

When an RNN is unrolled for $T$ time steps, the total loss $\mathcal{L}$ is the sum of losses at each time step: $\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t$.

To compute the gradient with respect to the recurrent matrix $\mathbf{W}_{hh}$, we must use the Multivariable Chain Rule across time.

Because $\mathbf{W}_{hh}$ was used at step $1$, step $2$, ..., step $t$, the gradient $\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$ must sum partial derivatives accumulated all the way back from step $t$ down to step $1$!

This creates a long chain of continuous matrix multiplications $\prod_{k=j+1}^t \mathbf{W}_{hh}^T$, directly exposing why vanilla RNNs suffer from vanishing gradients.


## 2. Key Definitions & Formal Terminology
- **Backpropagation Through Time (BPTT)**: The application of the backpropagation algorithm to an unrolled recurrent neural network, computing gradients by summing contributions across all time steps.
- **Truncated BPTT (TBPTT)**: A practical approximation that caps the backward unrolling horizon to a fixed number of steps $k$ (e.g., $k=20$), bounding compute and memory.
- **Temporal Jacobian Product**: The product of Jacobian matrices $\\prod_{k=j+1}^t \\frac{\\partial \\mathbf{h}_k}{\\partial \\mathbf{h}_{k-1}}$ that transmits gradient signals across temporal intervals.


## 3. Mathematical Formulations & Derivations
**Rigorous BPTT Derivation:**



Total Sequence Loss:

$$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$



The gradient with respect to $\mathbf{W}_{hh}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$



For a specific time step $t$, $\mathbf{h}_t$ depends on $\mathbf{W}_{hh}$ directly *and* indirectly through $\mathbf{h}_{t-1}$:

$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}} = \sum_{j=1}^t \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j} \cdot \frac{\partial^+ \mathbf{h}_j}{\partial \mathbf{W}_{hh}}$$



Where the temporal chain $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j}$ is a product of Jacobians:

$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j} = \prod_{k=j+1}^t \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=j+1}^t \mathbf{W}_{hh}^T \text{diag}\left( 1 - \mathbf{h}_k^2 \right)$$



And $\frac{\partial^+ \mathbf{h}_j}{\partial \mathbf{W}_{hh}} = \mathbf{h}_{j-1}^T$ represents the direct local derivative.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**BPTT Computational Backward Sweep:**

```

Loss L_T ------------> Loss L_{t} -------------> Loss L_1

    |                      |                        |

    v                      v                        v

dL/dh_T <--- W_hh^T -- dL/dh_t <--- W_hh^T --- dL/dh_1

    |                      |                        |

    v                      v                        v

dL/dW_hh (accumulates sum of products across all timesteps!)

```


## 5. Implementation Code Snippet
```python

import numpy as np



# BPTT Gradient computation loop for Vanilla RNN

def bptt_backward(inputs, targets, states, Why, Whh, Wxh):

    dWhh = np.zeros_like(Whh)

    dWxh = np.zeros_like(Wxh)

    dWhy = np.zeros_like(Why)

    dh_next = np.zeros((Whh.shape[0], 1))

    

    # Sweep backwards through time

    for t in reversed(range(len(inputs))):

        dy = outputs[t] - targets[t] # Loss derivative

        dWhy += np.dot(dy, states[t].T)

        

        # dh accumulated from output and future hidden state

        dh = np.dot(Why.T, dy) + dh_next

        # backprop through tanh: dtanh = (1 - h^2)

        da = (1 - states[t] ** 2) * dh

        

        dWxh += np.dot(da, inputs[t].T)

        h_prev = states[t-1] if t > 0 else np.zeros_like(states[0])

        dWhh += np.dot(da, h_prev.T)

        

        # Propagate to previous hidden state

        dh_next = np.dot(Whh.T, da)

        

    return dWxh, dWhh, dWhy

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does BPTT cause high memory usage on long sequences?
**Answer**:
BPTT requires storing the complete sequence of intermediate hidden state vectors $\mathbf{h}_0, \mathbf{h}_1, \dots, \mathbf{h}_T$ in GPU VRAM during the forward pass. For a sequence of length $T=1000$, memory usage is $1000\times$ higher than a single forward step. This is why **Truncated BPTT** is used to cap backprop to $20-50$ steps.

### Q2: How does BPTT reveal the mathematical origin of Vanishing and Exploding Gradients?
**Answer**:
The derivative chain contains the term $\prod_{k=j+1}^t \mathbf{W}_{hh}^T$. If the largest eigenvalue of $\mathbf{W}_{hh}$ is $\lambda < 1$, $\lambda^{t-j} \to 0$ exponentially as temporal distance $(t - j)$ grows, causing vanishing gradients. If $\lambda > 1$, $\lambda^{t-j} \to \infty$, causing exploding gradients.

### Q3: What is Truncated BPTT (TBPTT)?
**Answer**:
An engineering compromise where forward propagation runs across the entire sequence, but backpropagation stops after a fixed number of steps $k_1$, updating weights periodically every $k_2$ steps. This prevents VRAM exhaustion and stabilizes training.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Total gradient is the sum across all time steps: $\\sum_{t=1}^T \\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{W}}$.
- Long-range gradient contains $\\prod \\mathbf{W}_{hh}^T$, leading to exponential growth or decay.
- Truncated BPTT bounds memory and compute.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=OvCz1acvt-k)
- [Lecture Video](https://www.youtube.com/watch?v=OvCz1acvt-k)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
