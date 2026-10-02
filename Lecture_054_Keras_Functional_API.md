# Lecture 054: Keras Functional API | Building Non-Linear Neural Networks

> **CampusX 100 Days of Deep Learning** | Video ID: `OvQQP1QVru8` | Duration: 36m 12s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=OvQQP1QVru8) | **Transcript Status**: Available (hi, 3243 words)

---

## 1. Executive Summary & Core Intuition
Up to this point, models were built using `Sequential()`, which assumes a linear, single-input, single-output pipeline of stacked layers.

However, modern architectures require complex, non-linear computational graphs:

1. **Multi-Input Models:** Combining tabular metadata (age, clinical history) with image data (chest X-ray) to predict disease.

2. **Multi-Output Models:** Predicting both a bounding box coordinates (regression) and class label (classification) simultaneously from one image.

3. **Residual / Skip Connections:** Directly adding an earlier layer tensor to a later layer tensor ($x + F(x)$), as in ResNet.

4. **Shared Layer Models:** Siame networks comparing two signatures or faces using the exact same weight matrix.

The **Keras Functional API** treats layers as callable functions that accept and return tensors: `output_tensor = Layer(parameters)(input_tensor)`.


## 2. Key Definitions & Formal Terminology
- **Keras Functional API**: An architectural framework for defining complex Directed Acyclic Graph (DAG) deep learning models with non-linear topologies, shared layers, and multiple inputs/outputs.
- **Residual Skip Connection**: An architectural bypass where input tensor $x$ is added element-wise to transformed tensor $F(x)$: `layers.add([x, F(x)])`.
- **Multi-Task Learning**: Training a single unified model with multiple loss functions to simultaneously perform multiple distinct prediction tasks, sharing intermediate representations.


## 3. Mathematical Formulations & Derivations
**Multi-Loss Objective Function Formulation:**

For a multi-output network predicting classification target $y_{class}$ and regression target $y_{reg}$:



$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{class}(y_{class}, \hat{y}_{class}) + \lambda_2 \mathcal{L}_{reg}(y_{reg}, \hat{y}_{reg})$$



Where $\lambda_1, \lambda_2$ are loss weights balancing the gradient magnitudes of disparate tasks.

Total parameter gradient is the weighted sum:

$$\nabla_{\mathbf{W}} \mathcal{L}_{total} = \lambda_1 \nabla_{\mathbf{W}} \mathcal{L}_{class} + \lambda_2 \nabla_{\mathbf{W}} \mathcal{L}_{reg}$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Functional Multi-Input / Multi-Output DAG Architecture:**

```

Image Input (128x128x3) ----> [ Conv + Pool ] -------\

                                                      ---> [ Concatenate ] ---> [ Dense ] ---> Output 1: Gender (Sigmoid)

Tabular Input (10,)     ----> [ Dense(16) ]   -------/                                     ---> Output 2: Age (Linear)

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# 1. Residual Block using Functional API

def residual_block(input_tensor, filters):

    # Main path

    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)

    # Skip connection (addition)

    out = layers.add([x, input_tensor])

    out = layers.Activation('relu')(out)

    return out



# 2. Multi-Output Model

input_img = layers.Input(shape=(64, 64, 3), name='face_input')

x = layers.Conv2D(32, (3, 3), activation='relu')(input_img)

x = layers.MaxPooling2D()(x)

x = layers.Flatten()(x)

shared_features = layers.Dense(64, activation='relu')(x)



# Two task heads

gender_output = layers.Dense(1, activation='sigmoid', name='gender_out')(shared_features)

age_output = layers.Dense(1, activation='linear', name='age_out')(shared_features)



model = models.Model(inputs=input_img, outputs=[gender_output, age_output])

model.compile(

    optimizer='adam',

    loss={'gender_out': 'binary_crossentropy', 'age_out': 'mean_squared_error'},

    loss_weights={'gender_out': 1.0, 'age_out': 0.01}

)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: When MUST you use the Functional API instead of `Sequential()` in Keras?
**Answer**:
Whenever the model architecture is not a strictly linear chain of single-input, single-output layers. Specific cases include: (1) Residual skip connections (ResNets), (2) Inception modules with parallel branches, (3) Multiple inputs or multiple outputs, and (4) Shared layers (Siamese networks).

### Q2: What is the difference between `layers.add()` and `layers.concatenate()` in the Functional API?
**Answer**:
`layers.add([a, b])` performs element-wise addition ($a + b$), requiring tensors $a$ and $b$ to have identical shapes; channel count remains unchanged. `layers.concatenate([a, b], axis=-1)` stacks tensors along the specified axis, combining their channels: shape $(H, W, C_1)$ and $(H, W, C_2)$ become $(H, W, C_1 + C_2)$.

### Q3: Why are `loss_weights` critical when training multi-output models?
**Answer**:
Different loss functions operate on vastly different numerical scales. An MSE loss on age prediction might be $150.0$, while a BCE loss on gender prediction is $0.4$. Without loss weighting, the MSE gradient will overpower the network, causing the model to optimize age while completely ignoring gender.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Functional syntax: `tensor_out = Layer()(tensor_in)`.
- Use `add` for residual skips; use `concatenate` for parallel feature pooling.
- Multi-task learning requires calibrated `loss_weights`.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=OvQQP1QVru8)
- [Lecture Video](https://www.youtube.com/watch?v=OvQQP1QVru8)
- [Colab Notebook](https://colab.research.google.com/drive/1uCHf6hoLR1a-46RznVjqnhVZNechF0fz?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
