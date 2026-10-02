# Lecture 043: Padding & Strides in CNN | Dimension Arithmetic

> **CampusX 100 Days of Deep Learning** | Video ID: `btWE6SsdDZA` | Duration: 30m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=btWE6SsdDZA) | **Transcript Status**: Available (hi, 3554 words)

---

## 1. Executive Summary & Core Intuition
When convolving an image of size $N \times N$ with a $K \times K$ kernel, two problems arise:

1. **Shrinking Feature Maps:** Each convolution shrinks the spatial dimensions by $(K - 1)$. For a $32 \times 32$ image and $3 \times 3$ kernel, dimensions shrink to $30 \times 30$, then $28 \times 28$, rapidly reducing spatial resolution to $0 \times 0$ after a few layers.

2. **Border Pixel Information Loss:** Corner and edge pixels participate in very few receptive fields compared to central pixels, discarding critical perimeter visual information.

**Padding ($P$)** solves this by surrounding the image border with rows/columns of zeros:

- **Valid Padding ($P=0$):** No padding; dimensions shrink.

- **Same Padding:** Adds sufficient zero padding so that output spatial dimensions exactly equal input dimensions ($H_{out} = H_{in}$).

**Stride ($S$)** is the step size the kernel moves at each hop. A stride $S > 1$ downsamples the image, replacing pooling.


## 2. Key Definitions & Formal Terminology
- **Padding ($P$)**: The number of extra pixels added to the outer boundaries of an image tensor (typically zero-padding).
- **Valid Padding**: Convolution without padding ($P=0$); kernel only visits fully valid interior pixels.
- **Same Padding**: Padding chosen such that when stride $S=1$, the output feature map has the identical spatial height and width as the input.
- **Stride ($S$)**: The step size (in pixels) by which the convolution filter slides across the horizontal and vertical spatial dimensions.


## 3. Mathematical Formulations & Derivations
**The Universal Output Dimension Formula:**

For an input of spatial size $N \times N$, kernel size $K$, padding $P$, and stride $S$:



$$\text{Output Dimension } M = \left\lfloor \frac{N - K + 2P}{S} \right\rfloor + 1$$



For rectangular dimensions $(H_{in}, W_{in})$:

$$H_{out} = \left\lfloor \frac{H_{in} - K_h + 2P_h}{S_h} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} - K_w + 2P_w}{S_w} \right\rfloor + 1$$



**Formula for 'Same' Padding (when $S=1$):**

We require $M = N \implies N - K + 2P + 1 = N \implies 2P = K - 1$:

$$P = \frac{K - 1}{2}$$

*(Notice why kernel sizes $K$ are almost always chosen to be ODD numbers: $3, 5, 7$! If $K$ is odd, $K-1$ is even, allowing symmetric integer padding $P$.)*

- For $K=3 \implies P = \frac{3-1}{2} = 1$

- For $K=5 \implies P = \frac{5-1}{2} = 2$

- For $K=7 \implies P = \frac{7-1}{2} = 3$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Visualizing Same Padding ($P=1$ for $3 \times 3$ Kernel):**

```

0  0  0  0  0

0 [x  x  x] 0  <-- Added 1-pixel border of zeros on all sides.

0 [x  x  x] 0      Allows the 3x3 kernel center to visit

0 [x  x  x] 0      every original corner and edge pixel!

0  0  0  0  0

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



# Valid Padding: 28x28 -> (28 - 3)/1 + 1 = 26x26

conv_valid = layers.Conv2D(32, (3, 3), padding='valid', input_shape=(28, 28, 1))



# Same Padding: 28x28 -> 28x28 (auto pads P=1)

conv_same = layers.Conv2D(32, (3, 3), padding='same', input_shape=(28, 28, 1))



# Strided Convolution (S=2): 28x28 -> floor((28 - 3 + 2)/2) + 1 = 14x14

conv_strided = layers.Conv2D(32, (3, 3), strides=2, padding='same', input_shape=(28, 28, 1))

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Calculate the output shape of a $64 \times 64$ image convolved with a $5 \times 5$ filter, stride $S=2$, and padding $P=2$.
**Answer**:
Using the formula: $M = \\lfloor \\frac{N - K + 2P}{S} \\rfloor + 1 = \\lfloor \\frac{64 - 5 + 2(2)}{2} \\rfloor + 1 = \\lfloor \\frac{64 - 5 + 4}{2} \\rfloor + 1 = \\lfloor \\frac{63}{2} \\rfloor + 1 = 31 + 1 = 32$. Output shape is $32 \\times 32$.

### Q2: Why are convolutional kernels almost exclusively odd numbers ($3 \\times 3, 5 \\times 5$)?
**Answer**:
Two reasons: (1) Odd kernels have an unambiguous center pixel $(x, y)$, providing a natural coordinate anchor for the output feature map. (2) Odd kernels allow symmetric padding $P = \\frac{K-1}{2}$. Even kernels ($2 \\times 2, 4 \\times 4$) require asymmetric padding (e.g., 1 pixel on left, 2 on right), distorting spatial geometry.

### Q3: How does Strided Convolution ($S > 1$) relate to Pooling?
**Answer**:
Both perform spatial downsampling. However, Pooling is fixed and unlearnable (e.g., take max or average). Strided convolution is fully learnable—the network learns the optimal downsampling filter weights via backpropagation (Springenberg et al., 'All Convolutional Net').

## 7. Crucial Exam Takeaways & Common Pitfalls
- Formula to memorize: $M = \\lfloor \\frac{N - K + 2P}{S} \\rfloor + 1$.
- Same padding: $P = (K - 1) / 2$ (requires odd kernel size).
- Valid padding has $P=0$.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=btWE6SsdDZA)
- [Lecture Video](https://www.youtube.com/watch?v=btWE6SsdDZA)
- [Colab Notebook](https://colab.research.google.com/drive/1HBMLctcBnhvV6Rj62Zc8eAXERQw54l2H?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
