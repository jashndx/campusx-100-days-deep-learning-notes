# Lecture 079: Layer Normalization in Transformers | LayerNorm vs BatchNorm

> **CampusX 100 Days of Deep Learning** | Video ID: `qti0QPdaelg` | Duration: 29m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=qti0QPdaelg) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
Why do Transformers use **Layer Normalization (LayerNorm)** instead of **Batch Normalization (BatchNorm)**?

In Computer Vision (CNNs), Batch Normalization computes statistics across the mini-batch dimension: for a feature channel, it computes the mean across all $M$ images in the batch.

In Natural Language Processing (Transformers), this fails for two major reasons:

1. **Variable Sequence Lengths:** Different sentences have different lengths. Computing a batch mean across time step $t=150$ when only 2 sentences in a 32-sample batch are that long produces extremely noisy, unstable statistics.

2. **Dependency on Batch Size:** BatchNorm breaks when batch size is small ($B=1$ during inference).

**Layer Normalization (Ba, Kiros, Hinton, 2016)** normalizes **across the feature dimensions for each individual token independently**!

It requires no running statistics, is completely independent of batch size, works identically during training and testing, and provides rock-solid stability for Transformer training.


## 2. Key Definitions & Formal Terminology
- **Layer Normalization (LayerNorm)**: A normalization technique that normalizes inputs across features within a single data instance, rather than across instances in a batch.
- **Pre-LN vs Post-LN**: Architectural placement of LayerNorm: Post-LN places normalization after the residual addition ($x = \text{LN}(x + \text{Sublayer}(x))$); Pre-LN places it before ($\text{Sublayer}(\text{LN}(x)) + x$), which drastically improves gradient stability in deep models.
- **Feature Dimension Normalization**: Normalizing across the $d_{model}$ vector of a single token at a single time step: $\mu = \frac{1}{d} \sum_{i=1}^d x_i$.


## 3. Mathematical Formulations & Derivations
**Mathematical Comparison: BatchNorm vs LayerNorm:**



Let input tensor be $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$ (Batch $B$, Timesteps $T$, Hidden Features $D$).



1. **Batch Normalization (for fixed feature $d$, over $B$ and $T$):**

   $$\mu_d = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T X_{b, t, d}, \quad \sigma_d^2 = \frac{1}{B \cdot T} \sum_{b=1}^B \sum_{t=1}^T (X_{b, t, d} - \mu_d)^2$$

   *Normalized along vertical batch/sequence axis!*



2. **Layer Normalization (for single sample $b$ at time $t$, over feature dimension $D$):**

   $$\mu_{b, t} = \frac{1}{D} \sum_{i=1}^D X_{b, t, i}$$

   $$\sigma_{b, t}^2 = \frac{1}{D} \sum_{i=1}^D (X_{b, t, i} - \mu_{b, t})^2$$

   $$\hat{X}_{b, t, i} = \frac{X_{b, t, i} - \mu_{b, t}}{\sqrt{\sigma_{b, t}^2 + \epsilon}}$$

   $$y_{b, t, i} = \gamma_i \hat{X}_{b, t, i} + \beta_i$$

   *Normalized along horizontal feature axis for that specific token!*


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Axis of Normalization Visual Diagram:**

```

Tensor Shape: [ Batch B,  Time T,  Features D ]



Batch Normalization:                   Layer Normalization:

   +---------+                            +---------+

  /         /| (Down batch axis)         /         /|

 +---------+ |                          +---------+ |

 | [ * ]   | |                          | [*****] | |  <-- Normalizes across

 | [ * ]   | |                          |         | |      feature vector of

 | [ * ]   | +                          |         | +      ONE token!

 +---------+                            +---------+

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



# LayerNorm normalizes across the last axis (-1) by default

ln = layers.LayerNormalization(epsilon=1e-6)



# Input: (batch=32, seq_len=100, embed_dim=512)

x = tf.random.normal((32, 100, 512))

out = ln(x)

# Every single token vector now has mean=0.0 and var=1.0!

print("Mean of token 0:", tf.reduce_mean(out[0, 0, :]).numpy()) # ~ 0.0

print("Var of token 0:", tf.math.reduce_variance(out[0, 0, :]).numpy()) # ~ 1.0

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does Layer Normalization behave identically during training and testing, unlike Batch Normalization?
**Answer**:
BatchNorm computes statistics across batch partners during training, but must switch to frozen exponential moving averages during testing. LayerNorm computes mean and variance strictly across the features of the current token within the current sample. It has zero dependency on batch partners, making training and inference mathematically identical.

### Q2: Why is Pre-LN preferred over Post-LN in modern Large Language Models (like GPT-3, LLaMA)?
**Answer**:
In original Vaswani (Post-LN), gradients passing through residual connections must pass through LayerNorm, causing gradient magnitudes to grow or shrink unpredictably near the output, requiring a delicate learning rate warm-up. In Pre-LN, the residual shortcut ($x + \text{Sublayer}(\text{LN}(x))$) forms an uninterrupted identity highway where gradients flow unscaled directly from output to input, enabling training of models with hundreds of layers without warm-up.

### Q3: What are the trainable parameters of a LayerNormalization layer in a 512-dim Transformer?
**Answer**:
Two vectors of shape $(512,)$: the learnable scale parameter $\boldsymbol{\gamma}$ (initialized to 1) and the shift parameter $\boldsymbol{\beta}$ (initialized to 0). Total parameters = $512 + 512 = 1,024$ parameters.

## 7. Crucial Exam Takeaways & Common Pitfalls
- BatchNorm normalizes across batch; LayerNorm normalizes across feature dimension $D$.
- LayerNorm has zero dependency on batch size and works identically at inference.
- Pre-LN stabilizes gradient flow compared to original Post-LN.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=qti0QPdaelg)
- [Lecture Video](https://www.youtube.com/watch?v=qti0QPdaelg)
- [Official Course Notes](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
