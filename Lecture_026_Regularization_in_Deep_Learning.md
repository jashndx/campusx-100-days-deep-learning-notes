# Lecture 026: Regularization in Deep Learning | L1 vs L2 Weight Decay

> **CampusX 100 Days of Deep Learning** | Video ID: `4xRonrhtkzc` | Duration: 37m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=4xRonrhtkzc) | **Transcript Status**: Available (hi, 5692 words)

---

## 1. Executive Summary & Core Intuition
Regularization is any modification made to a learning algorithm that is intended to reduce its generalization error but not its training error.

Overfitting occurs when a neural network assigns excessively large numerical values to its weights $\mathbf{W}$. 

Large weights mean tiny input variations cause wild, violent swings in output predictions.

$L1$ and $L2$ Regularization add a penalty term to the loss function that penalizes large weights:

- **$L2$ Regularization (Ridge / Weight Decay):** Penalizes the sum of squares of weights ($\sum w^2$). It smoothly drives weights toward zero without forcing them to be exactly zero, keeping all features active while suppressing extreme reliance on any single connection.

- **$L1$ Regularization (Lasso):** Penalizes the sum of absolute values ($\sum |w|$). Geometrically, its diamond-shaped constraint boundaries drive weights to become strictly zero, producing **sparse representations** that perform automatic feature selection.


## 2. Key Definitions & Formal Terminology
- **Regularization**: Techniques used to prevent overfitting by penalizing model complexity or adding inductive constraints.
- **Weight Decay ($L2$ Regularization)**: Adding the squared Frobenius norm of the weight matrix $\\frac{\\lambda}{2} \\|\\mathbf{W}\\|_F^2$ to the loss function, causing weights to decay exponentially by a factor of $(1 - \\eta \\lambda)$ at each step.
- **Lasso Regularization ($L1$)**: Adding the $L1$ norm $\\lambda \\|\\mathbf{W}\\|_1$ to the loss function, driving insignificant weights to absolute zero.
- **Sparsity**: A structural condition where a high percentage of parameter values in a tensor are exactly zero, enabling compression and feature pruning.


## 3. Mathematical Formulations & Derivations
**Total Regularized Cost Function:**

$$\mathcal{J}(\mathbf{W}, \mathbf{b}) = \mathcal{L}(\mathbf{W}, \mathbf{b}) + \Omega(\mathbf{W})$$



**1. $L2$ Regularization (Weight Decay):**

$$\Omega_{L2}(\mathbf{W}) = \frac{\lambda}{2m} \sum_{l=1}^L \|\mathbf{W}^{[l]}\|_F^2 = \frac{\lambda}{2m} \sum_{l=1}^L \sum_{j} \sum_{k} (w_{jk}^{[l]})^2$$



Gradient with respect to $\mathbf{W}^{[l]}$:

$$\nabla_{\mathbf{W}^{[l]}} \mathcal{J} = \nabla_{\mathbf{W}^{[l]}} \mathcal{L} + \frac{\lambda}{m} \mathbf{W}^{[l]}$$



Weight Update Rule:

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \left( \nabla_{\mathbf{W}^{[l]}} \mathcal{L} + \frac{\lambda}{m} \mathbf{W}^{[l]} \right) = \left( 1 - \frac{\eta \lambda}{m} \right) \mathbf{W}^{[l]} - \eta \nabla_{\mathbf{W}^{[l]}} \mathcal{L}$$

Notice that before subtracting the gradient, the weight is decayed by factor $\left(1 - \frac{\eta \lambda}{m}\right) < 1$. Hence the name **Weight Decay**!



**2. $L1$ Regularization (Lasso):**

$$\Omega_{L1}(\mathbf{W}) = \frac{\lambda}{m} \sum_{l=1}^L \sum_{j} \sum_{k} |w_{jk}^{[l]}|$$

$$\nabla_{\mathbf{W}^{[l]}} \mathcal{J} = \nabla_{\mathbf{W}^{[l]}} \mathcal{L} + \frac{\lambda}{m} \text{sign}(\mathbf{W}^{[l]})$$

Subtracts a constant $\frac{\eta \lambda}{m}$ towards zero at every single update, driving weights to exact zero.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Geometric Comparison of $L1$ vs $L2$:**

```

     L1 Constraint (Diamond)                   L2 Constraint (Circle)

             w2                                        w2

             ^                                         ^

             |    /\                                   |      /---\

             |   /  \                                  |     /     \

             |  /    \  <-- Corner at axis!            |    |   O   |

    ---------+-(------+-)------> w1           ---------+----+-------+------> w1

             |  \    /                                 |     \     /

             |   \  /                                  |      \---/

             |    \/                                   |

    Loss contours touch at w2=0 (Sparsity)     Loss contours touch smoothly

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, regularizers



# Applying L1, L2, and ElasticNet (L1 + L2) in Keras

model = tf.keras.Sequential([

    # L2 Regularization (Weight Decay)

    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(20,)),

    # L1 Regularization (Sparse)

    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.005)),

    # Elastic Net (L1 + L2)

    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.01)),

    layers.Dense(1, activation='sigmoid')

])

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does $L1$ regularization lead to sparse weights while $L2$ does not?
**Answer**:
Geometrically, the $L1$ norm ball has sharp corners (vertices) along the coordinate axes where one or more parameters equal zero. When the elliptical loss contours expand, they are statistically far more likely to intersect the constraint boundary at these sharp corners. In $L2$, the boundary is a smooth sphere with no corners, so weights are shrunk continuously without being forced to zero.

### Q2: Why do we generally NOT regularize the bias vector $\mathbf{b}$?
**Answer**:
Biases contain only $n^{[l]}$ parameters compared to $n^{[l-1]} \times n^{[l]}$ weights; regularizing biases introduces significant underfitting bias without providing meaningful variance reduction. Furthermore, weights govern the slope/curvature of decision boundaries, whereas biases merely shift the location.

### Q3: What is the difference between $L2$ Regularization and Weight Decay in Adam?
**Answer**:
Loshchilov & Hutter (2019, AdamW paper) showed that in adaptive gradient algorithms like Adam, standard $L2$ regularization gradient addition gets distorted by dividing by the moving average of squared gradients $\sqrt{v_t}$. True Weight Decay decouples the weight decay step from the gradient update ($\mathbf{W} \leftarrow \mathbf{W}(1 - \eta \lambda) - \text{AdamStep}$), which is why `AdamW` outperforms standard `Adam`.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $\mathbf{W} \leftarrow (1 - \frac{\eta \lambda}{m})\mathbf{W} - \eta \nabla \mathcal{L}$.
- $L1$ = Sparsity (feature selection); $L2$ = Small, distributed weights.
- Do NOT regularize biases.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=4xRonrhtkzc)
- [Lecture Video](https://www.youtube.com/watch?v=4xRonrhtkzc)
- [Colab Notebook](https://colab.research.google.com/drive/1PObj5KrXLDDmHjoJ1x0bVmxAFbif5s7q?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
