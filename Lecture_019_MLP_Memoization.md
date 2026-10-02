# Lecture 019: MLP Memoization | Dynamic Programming in Neural Networks

> **CampusX 100 Days of Deep Learning** | Video ID: `rW0eeTXas4k` | Duration: 21m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=rW0eeTXas4k) | **Transcript Status**: Available (en-US, 3565 words)

---

## 1. Executive Summary & Core Intuition
This lecture establishes the direct theoretical equivalence between Backpropagation and Dynamic Programming (Memoization).

In algorithms, Dynamic Programming solves complex problems by breaking them into overlapping subproblems, computing each subproblem solution once, and storing (memoizing) it in a lookup table.

In an MLP:

- The forward pass memoizes the intermediate state activations $\mathbf{A}^{[l]}$ and pre-activations $\mathbf{Z}^{[l]}$.

- The backward pass memoizes the adjoint error signals $\boldsymbol{\delta}^{[l]}$.

To compute $\boldsymbol{\delta}^{[l-1]}$, the algorithm does not restart from the output layer; it simply queries the already-memoized value $\boldsymbol{\delta}^{[l]}$ and multiplies by the local transition matrix $\mathbf{W}^{[l]T}$.

This reduces the computational complexity from exponential $\mathcal{O}(2^L)$ to strictly linear $\mathcal{O}(L)$ in the number of layers.


## 2. Key Definitions & Formal Terminology
- **Memoization**: An optimization technique used primarily to speed up programs by storing the results of expensive function calls and returning the cached result when the same inputs occur again.
- **Overlapping Subproblems**: A problem property where the same recursive subproblems are encountered repeatedly rather than generating unique subproblems.
- **Optimal Substructure**: A problem property where an optimal solution to the overall problem contains within it optimal solutions to its subproblems.


## 3. Mathematical Formulations & Derivations
**Dynamic Programming Recurrence Relation in Backprop:**



Let subproblem $S(l)$ represent computing the error signal $\boldsymbol{\delta}^{[l]}$ for layer $l$.



**Base Case:**

$$\boldsymbol{\delta}^{[L]} = \nabla_{\mathbf{a}^{[L]}} \mathcal{L} \odot g'(\mathbf{z}^{[L]})$$



**Recursive Step (Reusing Memoized $S(l+1)$):**

$$\boldsymbol{\delta}^{[l]} = \left[ (\mathbf{W}^{[l+1]})^T \cdot \underbrace{\boldsymbol{\delta}^{[l+1]}}_{\text{Memoized Subproblem } S(l+1)} \right] \odot g'(\mathbf{z}^{[l]})$$



Because each $\boldsymbol{\delta}^{[l]}$ is retained in cache, computing all layer gradients requires exactly:

$$\sum_{l=1}^L \mathcal{O}(n^{[l]} \cdot n^{[l-1]}) = \mathcal{O}(\text{Total Parameters})$$

Without memoization (re-evaluating from output for each weight), the complexity explodes to $\mathcal{O}(2^L)$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Execution DAG with Memoization Cache Table:**

```

Layer:          [1]             [2]             [3] (Output)

Forward Cache:  Z^[1], A^[1]    Z^[2], A^[2]    Z^[3], A^[3]

Backward Cache: delta^[1] <---  delta^[2] <---  delta^[3]

```

During runtime, the memory overhead of the memoization table is $\sum_{l=1}^L (M \times n^{[l]})$, perfectly balancing time and space efficiency.


## 5. Implementation Code Snippet
```python

# Demonstrating memoization cache dictionary in backprop loop

cache = {}



# Forward pass with memoization

def forward_with_memo(X, params):

    cache['A0'] = X

    A = X

    for l in range(1, 4):

        cache[f'Z{l}'] = np.dot(A, params[f'W{l}']) + params[f'b{l}']

        A = np.maximum(0, cache[f'Z{l}'])

        cache[f'A{l}'] = A

    return A



# Backward pass reads directly from memoized table

def backward_with_memo(dAL, params):

    deltas = {}

    deltas['delta3'] = dAL # output delta

    # Linear recurrence: read previous delta, store new delta

    deltas['delta2'] = np.dot(deltas['delta3'], params['W3'].T) * (cache['Z2'] > 0)

    deltas['delta1'] = np.dot(deltas['delta2'], params['W2'].T) * (cache['Z1'] > 0)

    return deltas

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: How does Backpropagation satisfy the two core requirements of Dynamic Programming?
**Answer**:
1. **Optimal Substructure:** The gradient of the loss with respect to layer $l$ is directly constructed from the gradient with respect to layer $l+1$. 2. **Overlapping Subproblems:** The sensitivity $\\boldsymbol{\\delta}^{[l+1]}$ is shared across all incoming connections from all neurons in layer $l$, making caching and reuse highly efficient.

### Q2: What would happen to training time if we disabled the forward-pass activation cache?
**Answer**:
To evaluate $g'(\\mathbf{z}^{[l]})$ and $\\mathbf{a}^{[l-1]}$, the network would have to re-execute forward propagation from layer $0$ up to layer $l$ for every single weight update, increasing computational complexity by an order of magnitude.

### Q3: What is Gradient Checkpointing?
**Answer**:
Gradient Checkpointing is a memory-saving compromise between recomputation and full memoization. Instead of caching all layer activations in VRAM, it caches only a subset of 'checkpoint' layers and recomputes intermediate activations on-the-fly during the backward pass, trading 20% compute time for 60-80% memory reduction.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Backpropagation is Dynamic Programming applied to the chain rule.
- Space-time trade-off: Caching activations costs VRAM but keeps time complexity $\\mathcal{O}(N_{params})$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=rW0eeTXas4k)
- [Lecture Video](https://www.youtube.com/watch?v=rW0eeTXas4k)

---
