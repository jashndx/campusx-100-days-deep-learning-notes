# Lecture 034: SGD with Momentum Explained | Physics Intuition & Animations

> **CampusX 100 Days of Deep Learning** | Video ID: `vVS4csXRlcQ` | Duration: 32m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=vVS4csXRlcQ) | **Transcript Status**: Available (en-US, 5223 words)

---

## 1. Executive Summary & Core Intuition
SGD with Momentum introduces the Newtonian physics of inertia into optimization.

Picture a heavy ball rolling down a hilly terrain. 

When rolling down a slope, the ball accelerates, building up momentum ($v$). 

If it encounters small bumps, local divots, or flat spots, its accumulated inertia carries it straight through.

In neural network ravines (where the surface is steep along the vertical axis but shallow along the horizontal axis toward the minimum):

- Standard SGD bounces back and forth across the steep walls, making zero forward progress.

- Momentum averages out the alternating vertical oscillations (which cancel to zero) while building up relentless velocity along the consistent horizontal direction!


## 2. Key Definitions & Formal Terminology
- **Momentum**: A method that helps accelerate SGD in the relevant direction and dampens oscillations by incorporating a fraction $\\gamma$ of the update vector of the past time step.
- **Velocity Vector ($v_t$)**: An internal state variable that accumulates exponentially decaying historical gradients, dictating parameter update direction and speed.
- **Momentum Coefficient ($\gamma$ or $\beta$)**: A hyperparameter (typically 0.9) acting as a friction coefficient that determines how much prior velocity persists.


## 3. Mathematical Formulations & Derivations
**Mathematical Formulation of SGD with Momentum:**



At iteration $t$, compute gradient: $\mathbf{g}_t = \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}^{(t)})$.



**Velocity Update:**

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \eta \mathbf{g}_t \quad (\text{or } \mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\mathbf{g}_t)$$



**Parameter Update:**

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \mathbf{v}_t$$



**Terminal Velocity in Constant Gradient:**

If gradient $\mathbf{g}$ remains constant across iterations:

$$\mathbf{v}_\infty = \beta \mathbf{v}_\infty + \eta \mathbf{g} \implies \mathbf{v}_\infty = \frac{\eta \mathbf{g}}{1 - \beta}$$

For $\beta = 0.9$, the terminal velocity is $\frac{1}{1 - 0.9} = 10 \times (\eta \mathbf{g})$.

The step size automatically accelerates by a factor of 10 in consistent directions!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Oscillation Cancellation Geometry:**

```

Without Momentum (SGD):                 With Momentum:

   ^                                       ^

w2 |  /\  /\  /\  /\ (Violent           w2 |

   |  \/  \/  \/  \/  oscillations)        |  =======> (Oscillations cancel,

   +-----------------------> w1            +-----------------------> w1

                                                    fast horizontal progress)

```


## 5. Implementation Code Snippet
```python

import numpy as np



# Implementation of SGD with Momentum from scratch

class SGDMomentum:

    def __init__(self, lr=0.01, beta=0.9):

        self.lr = lr

        self.beta = beta

        self.v = None



    def update(self, w, grad):

        if self.v is None:

            self.v = np.zeros_like(w)

        # Velocity update

        self.v = self.beta * self.v + self.lr * grad

        # Weight update

        w = w - self.v

        return w

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: How does Momentum help an optimizer escape shallow local minima?
**Answer**:
When the ball rolls down into a shallow local basin, its accumulated kinetic energy (velocity $\\mathbf{v}$) carries it up and over the opposing energy barrier, escaping the trap where standard SGD (which only looks at local gradient $\\mathbf{g}=0$) would halt.

### Q2: Why is $\\beta=0.9$ the standard default value?
**Answer**:
A $\\beta=0.9$ corresponds to averaging gradients over approximately $\\frac{1}{1-0.9} = 10$ steps, which strikes the ideal balance between damping high-frequency stochastic oscillations without introducing excessive lag when navigating sharp turns.

### Q3: Can Momentum cause optimization to overshoot the minimum?
**Answer**:
Yes. If momentum is too high ($\beta \to 1$) and friction is too low, the parameter ball can overshoot the minimum and oscillate back and forth around the optimum before finally settling.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Terminal velocity equation: $\\mathbf{v}_\\infty = \\frac{\\eta \\mathbf{g}}{1 - \\beta}$.
- Cancels orthogonal oscillations while accelerating along persistent gradients.
- Default $\\beta = 0.9$ provides a $10\\times$ acceleration factor.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=vVS4csXRlcQ)
- [Lecture Video](https://www.youtube.com/watch?v=vVS4csXRlcQ)

---
