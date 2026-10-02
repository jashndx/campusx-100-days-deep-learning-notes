# Lecture 015: Backpropagation in Deep Learning | Part 1 | The What

> **CampusX 100 Days of Deep Learning** | Video ID: `6M1wWQmcUjQ` | Duration: 32m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=6M1wWQmcUjQ) | **Transcript Status**: Available (hi, 8037 words)

---

## 1. Executive Summary & Core Intuition
Backpropagation (backward propagation of errors) is the engine that drives neural network training.

The central problem: after computing the loss $\mathcal{L}$ at the output layer, how much blame does an individual weight deep inside layer 2 or layer 1 bear for that error?

Backpropagation solves the 'credit assignment problem' by applying the Multivariable Chain Rule of calculus.

It transmits the error gradient backwards from the output layer to the input layer, calculating the partial derivative $\frac{\partial \mathcal{L}}{\partial w_{jk}^{[l]}}$ for every single trainable parameter.

With these derivatives in hand, gradient descent knows exactly which direction and step size to adjust each weight to decrease the loss.


## 2. Key Definitions & Formal Terminology
- **Backpropagation**: An efficient algorithm for computing the gradient of the loss function with respect to all weights and biases in a neural network by recursive application of the chain rule from output to input.
- **Credit Assignment Problem**: The fundamental challenge of determining how much each individual weight or neuron in a multi-layered hierarchy contributed to the overall prediction error at the output.
- **Error Signal / Delta ($\delta_j^{[l]}$)**: The sensitivity of the total loss to perturbations in the pre-activation $z_j^{[l]}$: $\delta_j^{[l]} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}}$.
- **Chain Rule**: The fundamental theorem of calculus for finding the derivative of composite functions: if $y = f(u)$ and $u = g(x)$, then $\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}$.


## 3. Mathematical Formulations & Derivations
**The Fundamental Chain Rule Formulation:**

For a weight $w_{jk}^{[l]}$ connecting neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$:



$$\frac{\partial \mathcal{L}}{\partial w_{jk}^{[l]}} = \frac{\partial \mathcal{L}}{\partial z_j^{[l]}} \cdot \frac{\partial z_j^{[l]}}{\partial w_{jk}^{[l]}}$$



Since $z_j^{[l]} = \sum_p w_{jp}^{[l]} a_p^{[l-1]} + b_j^{[l]}$, the local derivative is simply the input activation:

$$\frac{\partial z_j^{[l]}}{\partial w_{jk}^{[l]}} = a_k^{[l-1]}$$



Defining the error term $\delta_j^{[l]} \equiv \frac{\partial \mathcal{L}}{\partial z_j^{[l]}}$:

$$\frac{\partial \mathcal{L}}{\partial w_{jk}^{[l]}} = \delta_j^{[l]} a_k^{[l-1]}$$

$$\frac{\partial \mathcal{L}}{\partial b_j^{[l]}} = \delta_j^{[l]}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Computational Flow Comparison:**

```

Forward Pass:

Input (a^[0]) ---> [Linear Z^[1]] ---> [Activation a^[1]] ---> ... ---> Loss L

                      |                         |

                      v                         v

               Cache Z^[1]                 Cache a^[1]



Backward Pass:

dL/da^[L] <--- [dL/dZ^[L]] <--- [dL/da^[L-1]] <--- ... <--- dL/da^[0]

                     |

                     v

             dL/dW^[L] = delta^[L] * (a^[L-1])^T

```


## 5. Implementation Code Snippet
```python

import numpy as np



# Single neuron backprop step demonstration

def single_neuron_backward(dL_da, z, a_prev, activation_derivative):

    # da/dz

    da_dz = activation_derivative(z)

    # delta = dL/dz

    delta = dL_da * da_dz

    # Gradients

    dL_dw = delta * a_prev

    dL_db = delta

    dL_da_prev = delta # multiplied by weight

    return dL_dw, dL_db, dL_da_prev

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What is the difference between Gradient Descent and Backpropagation?
**Answer**:
Backpropagation is an *algorithm for calculating derivatives* (gradients $\\nabla_{\\mathbf{W}} \\mathcal{L}$) via the chain rule. Gradient Descent is an *optimization algorithm* that uses those calculated gradients to update parameters ($\\mathbf{W} \\leftarrow \\mathbf{W} - \\eta \\nabla_{\\mathbf{W}} \\mathcal{L}$). Backprop computes; Gradient Descent updates.

### Q2: Why is the error delta $\\delta_j^{[l]}$ defined with respect to pre-activation $z_j^{[l]}$ rather than post-activation $a_j^{[l]}$?
**Answer**:
Because $z_j^{[l]}$ is the linear sum where all incoming weights $w_{jk}^{[l]}$ converge. By finding $\\frac{\\partial \\mathcal{L}}{\\partial z_j^{[l]}}$, the gradients for all incoming weights to that neuron are obtained simply by multiplying by the respective preceding activations $a_k^{[l-1]}$.

### Q3: Why does backpropagation run backwards from output to input rather than forwards?
**Answer**:
Running backwards computes the gradient of a single scalar loss with respect to all $M$ parameters in a single reverse sweep (Reverse-Mode Automatic Differentiation, $\\mathcal{O}(M)$ complexity). Running forward would require computing the gradient of each parameter one at a time, requiring $M$ forward passes ($\\mathcal{O}(M^2)$ complexity).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Backpropagation computes gradients; Gradient Descent updates weights.
- Weight gradient is: $\\delta_j^{[l]} \\times a_k^{[l-1]}$ (Error times incoming activation).
- Reverse-mode is computationally efficient for scalar losses.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=6M1wWQmcUjQ)
- [Lecture Video](https://www.youtube.com/watch?v=6M1wWQmcUjQ)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
