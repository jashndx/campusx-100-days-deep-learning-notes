# Lecture 071: Introduction to Transformers | The Architecture That Changed AI

> **CampusX 100 Days of Deep Learning** | Video ID: `BjRVS2wTtcA` | Duration: 35m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=BjRVS2wTtcA) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
In 2017, Ashish Vaswani et al. at Google Brain and Google Research published "Attention Is All You Need", introducing the **Transformer**—the architecture that completely revolutionized AI and powers modern LLMs (GPT-4, Gemini, Claude, LLaMA).

The foundational paradigm shift:

Up to 2017, attention was merely an auxiliary mechanism bolted onto recurrent (LSTM/GRU) backbones.

The Transformer proposed an audacious thesis: **Recurrence is completely unnecessary. Discard recurrence entirely. Attention is ALL you need!**

Why discard recurrence?

1. Recurrence is inherently sequential: $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$, making GPU parallelization across time impossible.

2. In recurrent networks, information between token $1$ and token $100$ must traverse $100$ sequential hops, risking degradation.

In a Transformer, **every token connects directly to every other token in a single operational step ($\mathcal{O}(1)$ path length)** via Self-Attention!

All $T$ tokens are processed in parallel, unlocking massive GPU training scalability.


## 2. Key Definitions & Formal Terminology
- **Transformer**: A deep learning architecture relying entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned recurrent networks or convolutions.
- **Self-Attention (Intra-Attention)**: An attention mechanism relating different positions of a single sequence in order to compute a representation of the exact same sequence.
- **Sequential Computation Constraint**: The computational bottleneck of RNNs where time step $t$ cannot begin processing until the output of step $t-1$ has finished computing.
- **Path Length ($\mathcal{O}(1)$)**: The number of operational forward hops required for a signal to propagate between any two arbitrary positions in an input sequence.


## 3. Mathematical Formulations & Derivations
**Path Length and Complexity Comparison Table (Vaswani et al., 2017):**



Let $n$ be sequence length, $d$ be representation dimension, and $k$ be convolution kernel size.



| Layer Type | Complexity per Layer | Max Path Length | Sequential Operations |

| :--- | :--- | :--- | :--- |

| **Recurrent (RNN/LSTM)** | $\mathcal{O}(n \cdot d^2)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ (Sequential Bottleneck!) |

| **Convolutional (1D CNN)**| $\mathcal{O}(k \cdot n \cdot d^2)$ | $\mathcal{O}(\log_k(n))$ | $\mathcal{O}(1)$ (Parallel) |

| **Self-Attention** | $\mathcal{O}(n^2 \cdot d)$ | $\mathbf{\mathcal{O}(1)}$ | $\mathbf{\mathcal{O}(1)}$ (Fully Parallel!) |



Notice that for Self-Attention:

- Sequential operations: $\mathcal{O}(1)$ $\implies$ All tokens processed simultaneously across GPU cores!

- Max path length: $\mathcal{O}(1)$ $\implies$ Token 1 connects to Token 10,000 in a single direct step!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**High-Level Transformer Architectural Topology:**

```

Input Sequence  ---> [ Positional Encoding ] ---> [ N x Encoder Blocks ] ---\

                                                                             ---> Cross-Attention

Target Sequence ---> [ Positional Encoding ] ---> [ N x Decoder Blocks ] ---/

                                                         |

                                                  [ Linear Head ]

                                                         |

                                                  [ Softmax Probabilities ]

```

Both Encoder and Decoder consist of $N$ identical stacked layers (standard: $N=6$).


## 5. Implementation Code Snippet
```python

import tensorflow as tf



# In a Transformer, an entire batch of sequences is passed simultaneously:

# Input shape: (Batch Size, Sequence Length, Embedding Dim)

# No temporal for-loop! The entire 3D tensor is processed in one matrix pass!

X = tf.random.normal((32, 128, 512)) # 32 sentences, 128 words, 512 dimensions

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does the Transformer have a maximum path length of $\mathcal{O}(1)$ compared to $\mathcal{O}(n)$ in RNNs?
**Answer**:
In an RNN, for token 1 to communicate with token $n$, the information must sequentially pass through $n$ intermediate recurrent cell state transitions ($n$ steps). In Self-Attention, an attention weight is computed directly between every pair of positions $(i, j)$ via dot product in a single matrix operation, connecting any two tokens in exactly 1 hop.

### Q2: If Self-Attention is $\mathcal{O}(n^2 \cdot d)$, why is it faster to train than an RNN with $\mathcal{O}(n \cdot d^2)$?
**Answer**:
When sequence length $n$ is smaller than embedding dimension $d$ (e.g., $n=512, d=768$, typical in NLP), $n^2 \cdot d < n \cdot d^2$. More importantly, the RNN operations are *sequential* (preventing hardware parallelization), while the Transformer operations are *matrix multiplications* that fully saturate thousands of GPU tensor cores simultaneously.

### Q3: What is the major trade-off of the $\mathcal{O}(n^2)$ complexity in Transformers?
**Answer**:
Memory and compute scale quadratically with sequence length $n$. Doubling the context window from 2,048 to 4,096 tokens increases attention matrix memory by $4\times$. This quadratic bottleneck inspired efficient/linear attention variants (FlashAttention, Linformer).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Transformers eliminate recurrence entirely in favor of parallel Self-Attention.
- Sequential operations are $\mathcal{O}(1)$, unlocking massive GPU training parallelism.
- Path length between any two tokens is $\mathcal{O}(1)$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=BjRVS2wTtcA)
- [Lecture Video](https://www.youtube.com/watch?v=BjRVS2wTtcA)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
