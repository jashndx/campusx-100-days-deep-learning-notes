# Lecture 033: Exponentially Weighted Moving Average (EWMA)

> **CampusX 100 Days of Deep Learning** | Video ID: `jAqVuYJ8TP8` | Duration: 24m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=jAqVuYJ8TP8) | **Transcript Status**: Available (hi, 2789 words)

---

## 1. Executive Summary & Core Intuition
Before understanding Momentum, RMSProp, and Adam, one must master the mathematical foundation underlying all of them: the Exponentially Weighted Moving Average (EWMA).

A standard simple moving average across window $k$ requires storing the past $k$ data points in memory, which is computationally expensive for millions of parameters.

EWMA computes an exponentially smoothed average recursively using a single line of memory!

By introducing a decay factor $\beta \in [0, 1)$, each new estimate is a convex combination of the current observation and the previous estimate:

$v_t = \beta v_{t-1} + (1 - \beta) \theta_t$.

EWMA averages approximately over the past $\frac{1}{1 - \beta}$ time steps, providing effective noise smoothing with $\mathcal{O}(1)$ memory.


## 2. Key Definitions & Formal Terminology
- **Exponentially Weighted Moving Average (EWMA)**: A recursive statistical filter that applies weighting factors which decrease exponentially for older data points.
- **Decay Factor ($\beta$)**: A hyperparameter between 0 and 1 that governs the memory horizon; $\\beta=0.9$ averages over the last $\\approx 10$ steps, while $\\beta=0.99$ averages over $\\approx 100$ steps.
- **Bias Correction**: An adjustment applied to early EWMA estimates to eliminate the cold-start artifact caused by initializing $v_0 = 0$.


## 3. Mathematical Formulations & Derivations
**Recursive Definition of EWMA:**

$$v_t = \beta v_{t-1} + (1 - \beta) \theta_t, \quad v_0 = 0$$



Expanding the recurrence relation:

$$v_1 = (1 - \beta) \theta_1$$

$$v_2 = \beta (1 - \beta) \theta_1 + (1 - \beta) \theta_2$$

$$v_t = (1 - \beta) \sum_{i=1}^t \beta^{t-i} \theta_i$$



**Effective Window Horizon:**

Since $(1 - \epsilon)^{1/\epsilon} \approx \frac{1}{e} \approx 0.35$, weights decay to approximately $1/3$ of their original weight after $\frac{1}{1 - \beta}$ steps.

Thus, EWMA roughly averages over:

$$k \approx \frac{1}{1 - \beta} \text{ time steps}$$

- If $\beta = 0.9 \implies \frac{1}{1 - 0.9} = 10 \text{ steps}$.

- If $\beta = 0.98 \implies \frac{1}{1 - 0.98} = 50 \text{ steps}$.



**Bias Correction Formula:**

Because $v_0 = 0$, the sum of coefficients $(1 - \beta) \sum_{i=1}^t \beta^{t-i} = (1 - \beta^t) < 1$. 

To remove the downward initialization bias during early iterations:

$$\hat{v}_t = \frac{v_t}{1 - \beta^t}$$

As $t \to \infty$, $\beta^t \to 0$, so $1 - \beta^t \to 1$, making bias correction automatically fade out!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Weight Decay Curve of Past Observations:**

```

Weight on Observation

 ^

 | * (1 - beta)

 |   \

 |     \

 |       \

 |         * (1 - beta) * beta^k

 |           \_________________________

 0----------------------------------------> Age of Observation (t - i)

```


## 5. Implementation Code Snippet
```python

import numpy as np



def compute_ewma(data, beta=0.9, bias_correction=True):

    v = 0.0

    smoothed = []

    for t, theta in enumerate(data, 1):

        v = beta * v + (1 - beta) * theta

        if bias_correction:

            v_corrected = v / (1 - beta**t)

            smoothed.append(v_corrected)

        else:

            smoothed.append(v)

    return smoothed

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Bias Correction necessary in EWMA for early iterations?
**Answer**:
Because $v_0$ is initialized to 0. For example, if $\beta=0.98$ and $\theta_1 = 100$, uncorrected $v_1 = 0.98(0) + 0.02(100) = 2.0$, which drastically underestimates the true value. With bias correction: $\hat{v}_1 = \frac{2.0}{1 - 0.98^1} = \frac{2.0}{0.02} = 100.0$, yielding an accurate unbiased estimate immediately.

### Q2: Why is EWMA preferred over Simple Moving Average in deep learning optimizers?
**Answer**:
Simple Moving Average requires keeping an explicit buffer of the past $K$ gradient tensors in GPU memory, consuming immense VRAM for models with hundreds of millions of parameters. EWMA requires storing only a single tensor $v$, achieving $\mathcal{O}(1)$ memory overhead.

### Q3: What is the effect of setting $\beta$ too close to 1 (e.g., $\beta = 0.999$)?
**Answer**:
The curve becomes excessively smooth, but extremely sluggish to adapt to recent shifts in gradient direction, lagging far behind current optimization dynamics.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $v_t = \\beta v_{t-1} + (1-\\beta)\\theta_t$.
- Effective memory window: $k = \\frac{1}{1-\\beta}$.
- Bias correction: $\\hat{v}_t = \\frac{v_t}{1 - \\beta^t}$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=jAqVuYJ8TP8)
- [Lecture Video](https://www.youtube.com/watch?v=jAqVuYJ8TP8)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
