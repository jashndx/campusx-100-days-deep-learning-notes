# Lecture 045: Classic CNN Architecture | LeNet-5 Architecture Breakdown

> **CampusX 100 Days of Deep Learning** | Video ID: `ewsvsJQOuTI` | Duration: 35m 12s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=ewsvsJQOuTI) | **Transcript Status**: Available (hi, 2805 words)

---

## 1. Executive Summary & Core Intuition
LeNet-5, developed by Yann LeCun in 1998 for handwritten check digit recognition at Bell Labs, is the seminal blueprint for modern convolutional neural networks.

Understanding LeNet-5 layer-by-layer provides the master template for all subsequent architectures (AlexNet, VGG, ResNet).

The fundamental architectural motif established by LeNet-5:

$$\text{Input} \longrightarrow [\text{Conv} \to \text{Pool}] \longrightarrow [\text{Conv} \to \text{Pool}] \longrightarrow [\text{Flatten}] \longrightarrow [\text{FC} \to \text{FC} \to \text{Output}]$$

As signals traverse deeper into the network:

- Spatial dimensions ($H, W$) systematically shrink (via subsampling/pooling).

- Channel depth ($C$) systematically increases (learning richer, more abstract feature combinations).

- The total parameter footprint is dominated by the fully connected classification head.


## 2. Key Definitions & Formal Terminology
- **LeNet-5**: The landmark 7-layer convolutional network introduced by Yann LeCun et al. in 1998 that successfully read 10-20% of all bank checks across the United States.
- **Subsampling**: The 1998 terminology for average pooling with learnable coefficient and bias: $y = \tanh(w \cdot \text{avg}(x) + b)$.
- **Feature Extraction vs Classification Head**: The standard division of CNNs into a convolutional base (which extracts spatial features) and a dense classifier (which maps features to class probabilities).


## 3. Mathematical Formulations & Derivations
**Complete Layer-by-Layer Parametric Accounting of LeNet-5:**



1. **Input:** $32 \times 32$ Grayscale image ($1$ channel).

2. **Layer C1 (Convolution):** 6 filters of size $5 \times 5$, Stride 1, Padding 0.

   - Output Shape: $\left(\frac{32 - 5}{1}\right) + 1 = 28 \times 28 \times 6$.

   - Parameters: $(5 \times 5 \times 1 + 1) \times 6 = 26 \times 6 = \mathbf{156}$.

3. **Layer S2 (Subsampling / Pool):** $2 \times 2$ window, Stride 2.

   - Output Shape: $14 \times 14 \times 6$.

   - Parameters: $(1 \text{ weight} + 1 \text{ bias}) \times 6 = \mathbf{12}$.

4. **Layer C3 (Convolution):** 16 filters of size $5 \times 5$, Stride 1, Padding 0 (sparse connection scheme).

   - Output Shape: $\left(\frac{14 - 5}{1}\right) + 1 = 10 \times 10 \times 16$.

   - Parameters: $\mathbf{1,516}$.

5. **Layer S4 (Subsampling / Pool):** $2 \times 2$ window, Stride 2.

   - Output Shape: $5 \times 5 \times 16$.

   - Parameters: $\mathbf{32}$.

6. **Layer C5 (Convolution / Flatten):** 120 filters of size $5 \times 5$, Stride 1.

   - Output Shape: $\left(\frac{5 - 5}{1}\right) + 1 = 1 \times 1 \times 120$.

   - Parameters: $(5 \times 5 \times 16 + 1) \times 120 = 401 \times 120 = \mathbf{48,120}$.

7. **Layer F6 (Fully Connected):** 84 neurons.

   - Parameters: $(120 + 1) \times 84 = \mathbf{10,164}$.

8. **Output Layer:** 10 classes (Euclidean Radial Basis Function).

   - Parameters: $84 \times 10 = \mathbf{840}$.



**Total Trainable Parameters:** $\approx \mathbf{60,840}$ parameters.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**LeNet-5 Architecture Diagram:**

```

Input(32x32) -> C1: 6@28x28 -> S2: 6@14x14 -> C3: 16@10x10 -> S4: 16@5x5 -> C5: 120@1x1 -> F6: 84 -> Out: 10

```

Notice that Layer C5 occupies over 79% of all parameters in the entire network ($48,120 / 60,840$)!


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



def build_lenet5():

    model = models.Sequential([

        # C1: 6 filters of 5x5, Tanh activation (original 1998 paper used Tanh)

        layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 1)),

        # S2: AveragePooling 2x2

        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        

        # C3: 16 filters of 5x5

        layers.Conv2D(16, (5, 5), activation='tanh'),

        # S4: AveragePooling 2x2

        layers.AveragePooling2D(pool_size=(2, 2), strides=2),

        

        # C5: Dense/Conv mapping to 120

        layers.Flatten(),

        layers.Dense(120, activation='tanh'),

        

        # F6: 84 units

        layers.Dense(84, activation='tanh'),

        

        # Output: 10 units (Softmax in modern implementation)

        layers.Dense(10, activation='softmax')

    ])

    return model



lenet = build_lenet5()

lenet.summary()

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why did LeNet-5 use a $32 \times 32$ input when MNIST digits are $28 \times 28$?
**Answer**:
LeCun padded the MNIST images with a 2-pixel border of zeros on all sides ($28 + 2 + 2 = 32$). This allowed boundary pixels and line strokes to pass through the centers of C1's $5 \times 5$ filters without clipping essential stroke endings.

### Q2: Why did C3 in original LeNet-5 not connect to all 6 channels of S2?
**Answer**:
Due to severe compute constraints in 1998, LeCun used a non-complete connection table (some filters connected to 3 channels, others to 4 or 6). This reduced parameter count and forced symmetry breaking among feature maps.

### Q3: Which part of LeNet-5 contains the majority of the parameters?
**Answer**:
The transition from the final convolutional feature maps to the fully connected layers (C5 and F6), which accounts for over 95% of the total network parameters.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Know the layer progression: Conv $\\to$ Pool $\\to$ Conv $\\to$ Pool $\\to$ FC $\\to$ Output.
- Total parameters in LeNet-5: $\\approx 60,000$.
- Modern adaptation replaces Tanh with ReLU and AveragePool with MaxPool.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=ewsvsJQOuTI)
- [Lecture Video](https://www.youtube.com/watch?v=ewsvsJQOuTI)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
