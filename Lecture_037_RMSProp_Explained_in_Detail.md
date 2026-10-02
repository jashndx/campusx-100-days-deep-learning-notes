# Lecture 037: RMSProp Explained in Detail | Root Mean Square Propagation

> **CampusX 100 Days of Deep Learning** | Video ID: `p0wSmKslWi0` | Duration: 31m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=p0wSmKslWi0) | **Transcript Status**: Available (en-US, 1569 words)

---

## 1. Executive Summary & Core Intuition
RMSProp (Root Mean Square Propagation) was invented by Geoffrey Hinton in Lecture 6 of his Coursera class (it was never officially published in a formal paper, yet became one of the most cited algorithms in deep learning).

RMSProp directly resolves the fatal flaw of AdaGrad.

Instead of summing all historical squared gradients from the beginning of time ($\sum_{1}^t g^2$), RMSProp uses an **Exponentially Weighted Moving Average (EWMA)** of squared gradients.

This gives the optimizer a 'sliding memory window' (typically past $\approx 10$ steps for $\beta=0.9$).

Recent gradient history dictates the current step size:

- If a parameter recently oscillated with massive gradients, its denominator expands, scaling down its step size.

- If a parameter recently entered a flat region with tiny gradients, its denominator contracts, scaling *up* its step size!

The learning rate never starves to zero, enabling stable convergence across thousands of training epochs.


## 2. Key Definitions & Formal Terminology
- **RMSProp**: An adaptive learning rate optimization algorithm that normalizes the gradient by an exponentially decaying average of squared gradients.
- **Discounting Factor ($\beta$ or $\rho$)**: The exponential decay hyperparameter (standard default $0.9$) controlling the effective memory horizon of squared gradients.
- **Root Mean Square (RMS)**: The square root of the arithmetic mean of the squares of values: $\\text{RMS}(g) = \\sqrt{\\mathbb{E}[g^2]}$.
- **Anisotropic Curvature Correction**: The ability of RMSProp to rescale steep and shallow directions of a loss ravine to identical step scales.


## 3. Mathematical Formulations & Derivations
**RMSProp Update Algorithm:**



At step $t$, compute mini-batch gradient: $\mathbf{g}_t = \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}^{(t)})$.



1. **Exponentially Decaying Average of Squared Gradients:**

   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \mathbf{g}_t^2$$

   *(Standard default: $\beta = 0.9$)*



2. **Parameter Update:**

   $$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \frac{\eta}{\sqrt{\mathbf{v}_t} + \epsilon} \odot \mathbf{g}_t$$

   *(where $\epsilon \approx 10^{-8}$ prevents division by zero)*



**Equi-gradient Step Property:**

Consider the ratio:

$$\frac{g_j}{\sqrt{v_{t, j}}} \approx \frac{g_j}{\sqrt{g_j^2}} = \frac{g_j}{|g_j|} = \text{sign}(g_j)$$

RMSProp approximately normalizes the update magnitude so that regardless of whether the raw gradient is $1000$ or $0.001$, the parameter takes a step of size proportional to $\pm \eta$!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**AdaGrad vs RMSProp Denominator Comparison:**

```

AdaGrad:  v_t = v_{t-1} + g_t^2                 (Unbounded monotonic growth -> LR freezes)

RMSProp:  v_t = 0.9 * v_{t-1} + 0.1 * g_t^2    (Bounded moving average -> LR stays adaptive)

```


## 5. Implementation Code Snippet
```python

import numpy as np



class RMSProp:

    def __init__(self, lr=0.001, beta=0.9, eps=1e-8):

        self.lr = lr

        self.beta = beta

        self.eps = eps

        self.v = None



    def update(self, w, grad):

        if self.v is None:

            self.v = np.zeros_like(w)

        # EWMA of squared gradients

        self.v = self.beta * self.v + (1 - self.beta) * (grad ** 2)

        # Rescaled update

        w = w - (self.lr / (np.sqrt(self.v) + self.eps)) * grad

        return w

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Geoffrey Hinton's RMSProp lecture on Coursera legendary in deep learning?
**Answer**:
Hinton introduced RMSProp in slide 29 of Lecture 6e of his 2012 Coursera course 'Neural Networks for Machine Learning'. Despite never being published in a peer-reviewed journal, it outperformed existing optimizers and was immediately adopted by TensorFlow, PyTorch, and deep learning practitioners worldwide.

### Q2: How does RMSProp handle ravines compared to Momentum?
**Answer**:
Momentum navigates ravines by accumulating directional velocity that cancels cross-ravine oscillations. RMSProp navigates ravines by *scaling*: it divides the steep vertical gradient by a huge number (shrinking oscillations) and divides the tiny horizontal gradient by a small number (amplifying forward progress).

### Q3: Why is $\epsilon \approx 10^{-8}$ necessary in the denominator?
**Answer**:
If a weight receives zero gradient for several iterations ($g_j = 0$), $v_{t, j}$ approaches zero. Attempting to divide by zero would trigger numerical `NaN` / `Inf` exceptions.

## 7. Crucial Exam Takeaways & Common Pitfalls
- RMSProp uses EWMA of squared gradients: $\\mathbf{v}_t = \\beta \\mathbf{v}_{t-1} + (1-\\beta)\\mathbf{g}_t^2$.
- Eliminates AdaGrad's learning rate starvation problem.
- Standard hyperparameters: $\\eta = 0.001, \\beta = 0.9, \\epsilon = 10^{-8}$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=p0wSmKslWi0)
- [Lecture Video](https://www.youtube.com/watch?v=p0wSmKslWi0)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
