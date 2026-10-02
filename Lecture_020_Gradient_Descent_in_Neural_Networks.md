# Lecture 020: Gradient Descent in Neural Networks | Batch vs Stochastic vs Mini Batch

> **CampusX 100 Days of Deep Learning** | Video ID: `7z6yXpYk7sw` | Duration: 34m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=7z6yXpYk7sw) | **Transcript Status**: Available (hi, 5827 words)

---

## 1. Executive Summary & Core Intuition
How should training data be supplied to the optimization algorithm?

This lecture compares the three fundamental flavors of Gradient Descent:

1. **Batch Gradient Descent:** Uses the entire dataset ($N$ samples) to compute one gradient update. Extremely stable and smooth convergence, but computationally prohibitive for large datasets and easily trapped in shallow local minima.

2. **Stochastic Gradient Descent (SGD):** Updates weights after evaluating every single sample ($B=1$). Extremely fast and memory efficient, but path oscillates wildly due to noisy sample-level gradients.

3. **Mini-Batch Gradient Descent:** The modern gold standard ($B \in [32, 256]$). Strikes the ideal balance: leverages GPU vectorization parallelism while providing moderate stochastic noise that helps escape saddle points and local minima.


## 2. Key Definitions & Formal Terminology
- **Batch Gradient Descent (BGD)**: Gradient descent where parameter updates are computed across the entire training dataset of $N$ instances simultaneously.
- **Stochastic Gradient Descent (SGD)**: Optimization where parameters are updated after computing the gradient on a single randomly sampled training instance.
- **Mini-Batch Gradient Descent**: Optimization where data is partitioned into small batches of size $B$ (typically powers of 2: 32, 64, 128), updating parameters after each mini-batch.
- **Epoch**: One complete pass of the optimization algorithm through the entire training dataset.
- **Iteration / Step**: A single parameter update step using one batch of data.


## 3. Mathematical Formulations & Derivations
**Update Rules Comparison:**



1. **Batch GD (1 update per epoch):**

   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \frac{1}{N} \sum_{i=1}^N \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}^{(t)})$$



2. **Pure SGD ($N$ updates per epoch):**

   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}^{(t)}) \quad (i \sim \text{Uniform}(1, N))$$



3. **Mini-Batch GD ($N/B$ updates per epoch, batch size $B$):**

   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \frac{1}{B} \sum_{i \in \mathcal{B}_k} \nabla_{\mathbf{w}} \mathcal{L}_i(\mathbf{w}^{(t)})$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Trade-Off Comparison Matrix:**



| Characteristic | Batch GD ($B=N$) | Stochastic GD ($B=1$) | Mini-Batch GD ($B \in [32, 256]$) |

| :--- | :--- | :--- | :--- |

| **Step Smoothness** | Monotonic, deterministic | High variance, erratic zig-zag | Balanced, controlled stochasticity |

| **GPU Parallelism** | High (if RAM permits) | Very Poor (memory bandwidth bound) | Optimal (fully utilizes SIMD cores) |

| **Escape Saddle Points**| Poor (gets trapped) | High (noise jolts parameters out) | Excellent |

| **Memory Footprint** | Extremely High $\mathcal{O}(N)$ | Minimal $\mathcal{O}(1)$ | Moderate $\mathcal{O}(B)$ |

| **Updates per Epoch** | 1 | $N$ | $\lceil N/B \rceil$ |


## 5. Implementation Code Snippet
```python

import numpy as np



# Generator for mini-batches

def get_mini_batches(X, y, batch_size=32, shuffle=True):

    N = X.shape[0]

    indices = np.arange(N)

    if shuffle:

        np.random.shuffle(indices)

    for start_idx in range(0, N, batch_size):

        end_idx = min(start_idx + batch_size, N)

        batch_idx = indices[start_idx:end_idx]

        yield X[batch_idx], y[batch_idx]



# Training loop

# for epoch in range(epochs):

#     for X_batch, y_batch in get_mini_batches(X_train, y_train, batch_size=64):

#         grads = compute_gradients(X_batch, y_batch)

#         update_parameters(grads, lr=0.01)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why are mini-batch sizes almost universally chosen as powers of 2 (e.g., 32, 64, 128, 256)?
**Answer**:
Hardware architectural alignment: GPU memory layouts, warp sizes (32 threads in NVIDIA CUDA architectures), and tensor core matrix tiles are structured around binary multiples. Choosing powers of 2 ensures optimal coalesced memory access and full hardware execution unit occupancy.

### Q2: Why does the stochastic noise in Mini-Batch GD help rather than hurt model generalization?
**Answer**:
Noise in mini-batch gradient estimates prevents the optimizer from settling into sharp, narrow local minima (which generalize poorly to unseen data). Instead, the noise jolts parameters toward broad, flat minima, which are empirically proven to generalize significantly better.

### Q3: How does the learning rate need to scale when increasing the batch size?
**Answer**:
According to the Linear Scaling Rule (Goyal et al., 2017), when increasing batch size $B$ by a factor of $k$, the learning rate $\eta$ should also be scaled up by $k$ (or $\sqrt{k}$ under warm-up regimes) to maintain equivalent optimization dynamics per epoch.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Mini-batch GD combines vectorization efficiency of Batch GD with noise benefits of SGD.
- Iterations per epoch = $\\lceil N / \\text{batch\\_size} \\rceil$.
- Always shuffle training data at the start of each epoch.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=7z6yXpYk7sw)
- [Lecture Video](https://www.youtube.com/watch?v=7z6yXpYk7sw)

---
