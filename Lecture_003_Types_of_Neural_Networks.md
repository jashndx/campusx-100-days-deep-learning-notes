# Lecture 003: Types of Neural Networks | History of Deep Learning | Applications

> **CampusX 100 Days of Deep Learning** | Video ID: `fne_UE7hDn0` | Duration: 28m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=fne_UE7hDn0) | **Transcript Status**: Available (en-US, 5891 words)

---

## 1. Executive Summary & Core Intuition
Different data modalities possess distinct structural symmetries (inductive biases). 

Tabular data requires general-purpose fully connected networks (ANN). 

Spatial data (images) exhibits translation invariance and local pixel correlation, demanding Convolutional Neural Networks (CNNs). 

Sequential and temporal data (audio, text, time-series) exhibits order and temporal context, demanding Recurrent Neural Networks (RNNs/LSTMs) and Transformers.

The history of deep learning is marked by cycles of hype and AI winters (from McCulloch-Pitts, Rosenblatt's Perceptron, Minsky-Papert's XOR critique, to Rumelhart's backprop rediscovery, LeCun's LeNet, and the 2012 AlexNet watershed).


## 2. Key Definitions & Formal Terminology
- **Artificial Neural Network (ANN / MLP)**: Fully connected feedforward architecture where each neuron in layer $l$ connects to every neuron in layer $l+1$; best suited for tabular and non-spatial data.
- **Convolutional Neural Network (CNN)**: Specialized neural network utilizing parameter sharing and spatial local receptive fields (convolution kernels) designed for 2D/3D grid-structured data like images.
- **Recurrent Neural Network (RNN)**: Architecture featuring cyclic hidden state connections, allowing information to persist across sequential time steps.
- **Inductive Bias**: The set of prior assumptions an algorithm uses to predict outputs of unseen inputs (e.g., spatial locality in CNNs, sequential ordering in RNNs).


## 3. Mathematical Formulations & Derivations
**Taxonomy of Structural Mappings:**

- **ANN (Fully Connected Layer):**

  $$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}), \quad \mathbf{W} \in \mathbb{R}^{M \times N}$$

- **CNN (Discrete 2D Cross-Correlation / Convolution):**

  $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$

- **RNN (Recurrent State Transition):**

  $$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Historical Milestones Timeline:**

1. **1943 - McCulloch-Pitts Neuron:** First mathematical abstraction of a biological neuron (binary threshold logic).

2. **1958 - Rosenblatt's Perceptron:** First learnable single-layer weight-updating machine.

3. **1969 - Minsky & Papert Critique:** Proved single-layer perceptrons cannot solve non-linear problems (XOR), triggering the first AI Winter.

4. **1986 - Rumelhart, Hinton & Williams:** Popularized Backpropagation for training multi-layer networks.

5. **1998 - Yann LeCun (LeNet-5):** Successful application of CNNs for handwritten digit recognition (check reading).

6. **2012 - AlexNet (Krizhevsky, Sutskever, Hinton):** Crushed ImageNet competition using GPUs and ReLU, igniting the modern Deep Learning revolution.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



# Architectural comparison in Keras

# 1. Dense (ANN)

ann_layer = layers.Dense(units=64, activation='relu')



# 2. Convolutional (CNN)

cnn_layer = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')



# 3. Recurrent (RNN)

rnn_layer = layers.SimpleRNN(units=64, activation='tanh')

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What caused the first AI Winter in 1969?
**Answer**:
Marvin Minsky and Seymour Papert published the book 'Perceptrons', proving mathematically that single-layer perceptrons could not compute the simple exclusive-OR (XOR) function, and pessimistically conjectured that extending them to multi-layers would be computationally intractable to train.

### Q2: Why is an ANN unsuitable for high-resolution images?
**Answer**:
Two primary reasons: (1) Parameter Explosion: A 1000x1000 RGB image has 3 million inputs; connecting to a hidden layer of 1000 neurons requires 3 billion weights, leading to immediate out-of-memory errors and extreme overfitting. (2) Loss of Spatial Structure: Flattening a 2D image into a 1D vector completely destroys 2D spatial locality and translation invariance.

### Q3: What is the key advantage of CNN parameter sharing?
**Answer**:
In CNNs, the same kernel filter is convolved across the entire spatial extent of the input image. This guarantees translation equivariance and drastically reduces the number of trainable parameters compared to fully connected layers.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Be able to draw and explain the historical timeline from McCulloch-Pitts (1943) to AlexNet (2012).
- State precisely why ANN is used for tabular, CNN for image, and RNN/Transformer for sequential data.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=fne_UE7hDn0)
- [Lecture Video](https://www.youtube.com/watch?v=fne_UE7hDn0)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
