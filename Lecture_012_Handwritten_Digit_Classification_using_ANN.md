# Lecture 012: Handwritten Digit Classification using ANN | MNIST Dataset

> **CampusX 100 Days of Deep Learning** | Video ID: `3xPT2Pk0Jds` | Duration: 38m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=3xPT2Pk0Jds) | **Transcript Status**: Available (hi, 4062 words)

---

## 1. Executive Summary & Core Intuition
This lecture tackles multi-class computer vision using an ANN on the benchmark MNIST dataset (70,000 $28 \times 28$ grayscale images of handwritten digits $0-9$).

Key pedagogical steps:

1. Understanding 2D image representations (matrices with pixel intensities $0-255$).

2. Flattening the $28 \times 28$ 2D matrix into a 784-dimensional 1D vector.

3. Normalizing pixel values by dividing by 255.0 to map inputs into $[0.0, 1.0]$.

4. Multi-class output architecture: 10 neurons with Softmax activation function.

5. Choosing between `categorical_crossentropy` (when targets are one-hot encoded) and `sparse_categorical_crossentropy` (when targets are integer class labels $0-9$).


## 2. Key Definitions & Formal Terminology
- **MNIST Dataset**: The Modified National Institute of Standards and Technology dataset consisting of 60,000 training and 10,000 testing $28 \\times 28$ grayscale handwritten digits.
- **Flattening**: Reshaping a multi-dimensional array (e.g., $(28, 28)$) into a single continuous 1D vector of length $28 \\times 28 = 784$.
- **Sparse Categorical Cross-Entropy**: A cross-entropy loss formulation that accepts integer class labels directly without requiring one-hot encoding matrices, conserving significant memory.


## 3. Mathematical Formulations & Derivations
**Categorical vs Sparse Categorical Cross-Entropy:**



Let true class label be $y \in \{0, 1, \dots, K-1\}$ and predicted probability vector be $\hat{\mathbf{y}} = [\hat{y}_0, \dots, \hat{y}_{K-1}]^T$:



1. **One-Hot Encoded (Categorical Cross-Entropy):**

   $$\mathbf{y}_{one\_hot} = [0, \dots, 1, \dots, 0]^T$$

   $$\mathcal{L}_{CCE} = -\sum_{k=0}^{K-1} y_k \log(\hat{y}_k)$$



2. **Integer Label (Sparse Categorical Cross-Entropy):**

   Since all terms in the summation are zero except where $k = y_{true}$:

   $$\mathcal{L}_{SCCE} = -\log(\hat{y}_{y_{true}})$$



Both yield mathematically identical gradients, but Sparse CCE saves $\mathcal{O}(N \times K)$ memory allocations.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**MNIST ANN Model Architecture:**

```

[Input: (28, 28)] 

       |

  [Flatten Layer] ---> Output: (784,)

       |

  [Dense(128, ReLU)] ---> Params: 784 * 128 + 128 = 100,480

       |

  [Dense(32, ReLU)] ---> Params: 128 * 32 + 32 = 4,128

       |

  [Dense(10, Softmax)] ---> Params: 32 * 10 + 10 = 330

```

Total Trainable Parameters: $100,480 + 4,128 + 330 = 104,938$.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# 1. Load MNIST

mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()



# 2. Pixel Normalization

X_train, X_test = X_train / 255.0, X_test / 255.0



# 3. Model Architecture

model = models.Sequential([

    layers.Flatten(input_shape=(28, 28)),

    layers.Dense(128, activation='relu'),

    layers.Dense(32, activation='relu'),

    layers.Dense(10, activation='softmax')

])



model.compile(optimizer='adam',

              loss='sparse_categorical_crossentropy',

              metrics=['accuracy'])



model.fit(X_train, y_train, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_acc * 100:.2f}%")

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why do we normalize pixel values from $[0, 255]$ to $[0, 1]$ before passing them to an ANN?
**Answer**:
Large input magnitudes (up to 255) cause the dot products $\\mathbf{z} = \\mathbf{X}\\mathbf{W} + \\mathbf{b}$ to explode, pushing activations into the saturated regions of activation functions (vanishing gradients) and creating an elongated, ill-conditioned loss surface where gradient descent oscillates wildly.

### Q2: When should one use `categorical_crossentropy` versus `sparse_categorical_crossentropy` in Keras?
**Answer**:
Use `categorical_crossentropy` when targets are one-hot encoded vectors (shape $(N, K)$). Use `sparse_categorical_crossentropy` when targets are integer scalars (shape $(N,)$). They calculate the exact same mathematical loss, but sparse CCE avoids generating large one-hot matrices.

### Q3: How do you extract the predicted class label from the 10 Softmax probability outputs?
**Answer**:
Using `np.argmax(probabilities, axis=1)`, which selects the index of the highest predicted probability.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Always divide image pixels by 255.0.
- Flatten layer has 0 trainable parameters—it only changes tensor shape.
- Softmax is used at output layer for multi-class classification.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=3xPT2Pk0Jds)
- [Lecture Video](https://www.youtube.com/watch?v=3xPT2Pk0Jds)
- [Colab Notebook](https://colab.research.google.com/drive/1SqETl3Zi1EEesdJfEv6_QimB-M-YjKGx?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
