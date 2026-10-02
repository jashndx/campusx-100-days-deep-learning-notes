# Lecture 004: What is a Perceptron? Perceptron Vs Neuron | Geometric Intuition

> **CampusX 100 Days of Deep Learning** | Video ID: `X7iIKPoZ0Sw` | Duration: 24m 50s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=X7iIKPoZ0Sw) | **Transcript Status**: Available (hi, 5927 words)

---

## 1. Executive Summary & Core Intuition
The Perceptron is the foundational building block of Artificial Neural Networks. 

Biologically inspired by the neuron: dendrites receive electrical signals ($x_i$), the cell body (soma) accumulates and sums weighted inputs ($\sum w_i x_i + b$), and the axon fires an action potential if the sum breaches a threshold.

Geometrically, a perceptron defines a linear hyperplane in $d$-dimensional space that separates two classes:

- In 2D space: a straight line ($w_1 x_1 + w_2 x_2 + b = 0$).

- In 3D space: a 2D plane ($w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$).

- In $n$-D space: an $(n-1)$-dimensional hyperplane ($\mathbf{w}^T \mathbf{x} + b = 0$).

Points on one side yield positive dot products (Class 1), while points on the other yield negative dot products (Class 0).


## 2. Key Definitions & Formal Terminology
- **Perceptron**: A binary linear classification algorithm invented by Frank Rosenblatt that maps real-valued input vectors to binary output decisions via a step activation function.
- **Hyperplane**: An affine subspace of dimension $k-1$ in a $k$-dimensional vector space that divides the space into two disconnected half-spaces.
- **Linear Separability**: A geometric property where two sets of points can be completely segregated by at least one flat hyperplane without misclassifying any instance.


## 3. Mathematical Formulations & Derivations
**Perceptron Forward Pass:**

Given input vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ and weight vector $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$ with bias $b$:



$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b$$



**Step Activation Function:**

$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$



**Geometric Distance to Decision Boundary:**

The signed perpendicular Euclidean distance from any point $\mathbf{x}_i$ to the separating hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ is:

$$d(\mathbf{x}_i) = \frac{\mathbf{w}^T \mathbf{x}_i + b}{\|\mathbf{w}\|_2} = \frac{\mathbf{w}^T \mathbf{x}_i + b}{\sqrt{\sum_{j=1}^d w_j^2}}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
```

   x1 ----(w1)----\

   x2 ----(w2)-----> [ Sum: z = w^T x + b ] ---> [ Step Function f(z) ] ---> Output y_hat in {0, 1}

   xd ----(wd)----/           ^

                             |

                           Bias b

```

**Decision Boundary Properties:**

- If $\mathbf{w}^T \mathbf{x} + b > 0 \implies \hat{y} = 1$ (Positive Half-Space).

- If $\mathbf{w}^T \mathbf{x} + b < 0 \implies \hat{y} = 0$ (Negative Half-Space).

- If $\mathbf{w}^T \mathbf{x} + b = 0 \implies$ Points lie exactly on the decision boundary.


## 5. Implementation Code Snippet
```python

import numpy as np



class Perceptron:

    def __init__(self, input_dim):

        self.weights = np.zeros(input_dim)

        self.bias = 0.0



    def predict(self, x):

        # Linear dot product

        z = np.dot(x, self.weights) + self.bias

        # Step activation function

        return 1 if z >= 0 else 0

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What is the geometric role of the bias term $b$ in a perceptron?
**Answer**:
The bias term shifts the decision boundary hyperplane away from the origin. Without a bias ($b=0$), the hyperplane is constrained to pass through the coordinate origin $(0, 0, \dots, 0)$, severely limiting its ability to separate linearly separable datasets that do not center at the origin.

### Q2: What does the weight vector $\mathbf{w}$ represent geometrically?
**Answer**:
The weight vector $\mathbf{w}$ is the normal (perpendicular) vector to the separating hyperplane. It points in the direction of the positive half-space where $\hat{y} = 1$.

### Q3: Why is the step function problematic for gradient descent?
**Answer**:
The standard Heaviside step function is non-differentiable at $z=0$ and has a derivative of zero everywhere else ($\frac{df}{dz} = 0 \ \forall z \neq 0$). Under gradient descent, gradients would vanish immediately, making backpropagation impossible.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Hyperplane equation: $\mathbf{w}^T \mathbf{x} + b = 0$.
- Weight vector $\mathbf{w}$ is orthogonal to the decision boundary line/plane.
- Perceptron can only solve strictly linearly separable problems.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=X7iIKPoZ0Sw)
- [Lecture Video](https://www.youtube.com/watch?v=X7iIKPoZ0Sw)

---
