# Lecture 017: Backpropagation Part 3 | The Why | Computational Graphs & Memoization

> **CampusX 100 Days of Deep Learning** | Video ID: `6xO-x8y0YSY` | Duration: 29m 50s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=6xO-x8y0YSY) | **Transcript Status**: Available (hi, 6305 words)

---

## 1. Executive Summary & Core Intuition
Why is backpropagation fundamentally designed the way it is?

This lecture looks under the hood through the lens of Computational Graphs and Dynamic Programming.

Any complex neural network is a Directed Acyclic Graph (DAG) of elementary mathematical operations.

If one attempted to compute derivatives by symbolic calculus, expressions would expand exponentially due to repeated sub-expressions (expression swell).

Backpropagation avoids this exponential catastrophe via **Dynamic Programming (Memoization)**: it computes the derivative at each node exactly once, caches it, and reuses it for all upstream dependencies.


## 2. Key Definitions & Formal Terminology
- **Computational Graph**: A directed acyclic graph (DAG) where nodes correspond to operations or variables, and directed edges represent data dependencies between inputs and outputs.
- **Forward Mode Automatic Differentiation**: Evaluates derivatives along with primal values from inputs to outputs; efficient when number of inputs is much smaller than outputs ($N_{in} \\ll N_{out}$).
- **Reverse Mode Automatic Differentiation**: Evaluates derivatives backwards from outputs to inputs; extraordinarily efficient for scalar objective functions with millions of parameters ($N_{in} \\gg N_{out} = 1$).
- **Expression Swell**: The exponential growth in the size of symbolic derivative mathematical expressions when computed naively without reusing intermediate subexpressions.


## 3. Mathematical Formulations & Derivations
**Computational Complexity Comparison:**



Let $N$ be the number of trainable parameters (e.g., $10^8$) and $f: \mathbb{R}^N \to \mathbb{R}$ be the scalar loss function.



1. **Numerical Differentiation (Finite Differences):**

   $$\frac{\partial \mathcal{L}}{\partial w_i} \approx \frac{\mathcal{L}(\mathbf{w} + \epsilon \mathbf{e}_i) - \mathcal{L}(\mathbf{w})}{\epsilon}$$

   Requires $N + 1$ forward passes: $\mathcal{O}(N)$ evaluations, computationally impossible for deep networks.



2. **Forward-Mode Autodiff:**

   Propagates tangent vectors $\dot{x}$. Requires $N$ forward passes: $\mathcal{O}(N)$ complexity.



3. **Reverse-Mode Autodiff (Backpropagation):**

   Propagates adjoint vectors $\bar{x} = \frac{\partial \mathcal{L}}{\partial x}$. 

   Computes gradients for ALL $N$ parameters in **1 single forward pass + 1 single backward pass**:

   $$\text{Time Complexity} \le 3 \times \text{Cost}(\text{Forward Pass})$$

   $$\mathcal{O}(1) \text{ passes with respect to } N!$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Node Computational Blueprint in Reverse-Mode:**

```

Forward:

   x --- \

          [ Operator: z = f(x, y) ] ---> z

   y --- /



Backward:

   dL/dx = (dL/dz) * (dz/dx) <--- \

                                  [ Operator: local grads ] <--- dL/dz (Adjoint)

   dL/dy = (dL/dz) * (dz/dy) <--- /

```

At any branching node where variable $u$ affects multiple downstream paths $v_1, v_2, \dots, v_k$:

$$\frac{\partial \mathcal{L}}{\partial u} = \sum_{j=1}^k \frac{\partial \mathcal{L}}{\partial v_j} \frac{\partial v_j}{\partial u}$$

Grades sum together at converging backward branches (Multivariable Chain Rule).


## 5. Implementation Code Snippet
```python

# Minimal Autograd Engine (Scalar Reverse-Mode)

class Value:

    def __init__(self, data, _children=()):

        self.data = data

        self.grad = 0.0

        self._backward = lambda: None

        self._prev = set(_children)



    def __add__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other))

        def _backward():

            self.grad += 1.0 * out.grad

            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out



    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other))

        def _backward():

            self.grad += other.data * out.grad

            other.grad += self.data * out.grad

        out._backward = _backward

        return out

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Reverse-Mode Autodiff chosen over Forward-Mode in Deep Learning?
**Answer**:
Deep Learning trains networks where $N_{parameters}$ is in millions or billions, but the loss is a single scalar ($N_{out} = 1$). Reverse-mode computes all parameter gradients in one reverse pass ($\mathcal{O}(1)$ passes). Forward-mode would require millions of forward passes ($\mathcal{O}(N_{params})$), which is computationally intractable.

### Q2: What is the 'Multivariate Chain Rule' node addition property in backpropagation?
**Answer**:
When a neuron's activation is fed into multiple subsequent neurons, its contribution splits across multiple computational paths. In the backward pass, the total derivative with respect to that neuron is the *sum* of the gradients flowing back from all its downstream branches: $\\frac{\\partial \\mathcal{L}}{\\partial x} = \\sum_j \\frac{\\partial \\mathcal{L}}{\\partial y_j} \\frac{\\partial y_j}{\\partial x}$.

### Q3: What is the memory trade-off of Reverse-Mode Automatic Differentiation?
**Answer**:
Reverse-mode requires caching all intermediate activations and computational graph nodes during the forward pass so they are available during the backward pass. This causes GPU VRAM consumption to scale linearly with network depth and batch size $\\mathcal{O}(L \\cdot M)$, leading to Out-Of-Memory (OOM) errors.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Backprop is Reverse-Mode Automatic Differentiation on a DAG.
- Gradients accumulate (sum) across branching paths.
- Memory complexity is $\\mathcal{O}(Layers \\times Batch)$, time complexity is $\\mathcal{O}(1)$ relative to parameter count.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=6xO-x8y0YSY)
- [Lecture Video](https://www.youtube.com/watch?v=6xO-x8y0YSY)

---
