# Lecture 083: Transformer Decoder Architecture | Complete Layer Breakdown

> **CampusX 100 Days of Deep Learning** | Video ID: `DI2_hrAulYo` | Duration: 44m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=DI2_hrAulYo) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
This lecture delivers the comprehensive, end-to-end breakdown of the **Transformer Decoder**.

While the Encoder has 2 sublayers per block, the Decoder has **3 sublayers per block**:

1. **Sublayer 1:** Masked Multi-Head Self-Attention (with look-ahead causal mask).

2. **Sublayer 2:** Cross-Attention (Multi-Head Attention over Encoder output).

3. **Sublayer 3:** Position-wise Feed-Forward Network (FFN).

Each of the 3 sublayers is enveloped by a Residual Skip Connection and Layer Normalization.

At the exit of the $N=6$ stacked Decoder blocks, the final representation passes through:

- A **Linear Projection Layer** that projects $d_{model}=512$ up to the entire target vocabulary size $V$ (e.g., $V = 32,000$ or $50,000$).

- A **Softmax layer** that produces a normalized probability distribution over all vocabulary tokens, from which the next token is sampled.


## 2. Key Definitions & Formal Terminology
- **Transformer Decoder**: The autoregressive generative sub-network of the Transformer that synthesizes output sequences token-by-token using causal self-attention and cross-attention.
- **Linear Projection Head**: A dense linear layer that maps the final decoder state vector $\mathbf{h} \in \mathbb{R}^{d_{model}}$ to logit scores over vocabulary size $V$: $\mathbf{z} \in \mathbb{R}^V$.
- **Weight Tying**: An optimization technique (Press & Wolf, 2017) where the target embedding matrix and the final linear projection weight matrix share the exact same parameters: $\mathbf{W}_{out} = \mathbf{E}_{target}^T$, saving tens of millions of parameters.


## 3. Mathematical Formulations & Derivations
**Complete Mathematical Step-by-Step of One Decoder Layer:**



Given previous decoder layer output $\mathbf{S}^{[l-1]}$ and encoder memory $\mathbf{H}_{enc}$:



**Sublayer 1 (Masked Self-Attention + Add & Norm):**

$$\mathbf{S}_{self} = \text{MultiHead}(\mathbf{S}^{[l-1]}, \mathbf{S}^{[l-1]}, \mathbf{S}^{[l-1]}, \text{mask}=\text{Causal})$$

$$\mathbf{S}^{(1)} = \text{LayerNorm}\left(\mathbf{S}^{[l-1]} + \mathbf{S}_{self}\right)$$



**Sublayer 2 (Cross-Attention + Add & Norm):**

$$\mathbf{S}_{cross} = \text{MultiHead}(\mathbf{Q}=\mathbf{S}^{(1)}, \mathbf{K}=\mathbf{H}_{enc}, \mathbf{V}=\mathbf{H}_{enc})$$

$$\mathbf{S}^{(2)} = \text{LayerNorm}\left(\mathbf{S}^{(1)} + \mathbf{S}_{cross}\right)$$



**Sublayer 3 (Position-wise FFN + Add & Norm):**

$$\mathbf{S}_{ffn} = \max\left(0, \mathbf{S}^{(2)}\mathbf{W}_1 + \mathbf{b}_1\right)\mathbf{W}_2 + \mathbf{b}_2$$

$$\mathbf{S}^{[l]} = \text{LayerNorm}\left(\mathbf{S}^{(2)} + \mathbf{S}_{ffn}\right)$$



**Final Output Layer (Above Layer 6):**

$$\text{Logits } \mathbf{z}_t = \mathbf{S}_t^{[6]} \mathbf{W}_{proj} + \mathbf{b}_{proj} \quad (\mathbf{W}_{proj} \in \mathbb{R}^{d_{model} \times V})$$

$$P(w_{t} \mid w_{<t}, \mathbf{X}) = \text{Softmax}(\mathbf{z}_t)$$


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Single Decoder Layer Architecture Block:**

```

Input Target Tokens: S^[l-1]

       |

  [ Masked Multi-Head Attention (Causal) ] ---> [ Add & Norm ] ---> S^(1)

                                                                       |

  [ Cross-Attention (Q: S^(1), K: H_enc, V: H_enc) ] -----------> [ Add & Norm ] ---> S^(2)

                                                                                         |

  [ Feed-Forward Network: Dense(2048) -> Dense(512) ] ----------> [ Add & Norm ] ---> S^[l]

```

Repeated across $N=6$ stacked layers $\to$ Linear Projection $\to$ Softmax.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



class TransformerDecoderLayer(layers.Layer):

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_rate=0.1):

        super().__init__()

        self.self_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)

        self.cross_mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads)

        self.ffn = tf.keras.Sequential([

            layers.Dense(d_ff, activation='relu'),

            layers.Dense(d_model)

        ])

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.ln3 = layers.LayerNormalization(epsilon=1e-6)



    def call(self, x, enc_output, training=False, causal_mask=None):

        # 1. Masked Self-Attention

        self_out = self.self_mha(query=x, value=x, key=x, attention_mask=causal_mask)

        x = self.ln1(x + self_out)

        

        # 2. Cross-Attention

        cross_out = self.cross_mha(query=x, value=enc_output, key=enc_output)

        x = self.ln2(x + cross_out)

        

        # 3. FFN

        ffn_out = self.ffn(x)

        x = self.ln3(x + ffn_out)

        return x

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Compare the sublayers of an Encoder block versus a Decoder block.
**Answer**:
An **Encoder block** has 2 sublayers: (1) Unmasked Bidirectional Self-Attention, and (2) FFN. A **Decoder block** has 3 sublayers: (1) Causal Masked Self-Attention, (2) Cross-Attention over encoder output, and (3) FFN. Both employ Add & Norm around every sublayer.

### Q2: Why does the Decoder contain significantly more parameters than the Encoder?
**Answer**:
Because of the extra Cross-Attention sublayer in every block ($1.05\text{M}$ extra parameters per layer $\times 6 = 6.3\text{M}$ params), plus the massive final linear classification projection head $\mathbf{W}_{proj} \in \mathbb{R}^{d_{model} \times V}$ which maps 512 dimensions to 32,000 vocabulary logits ($16.4\text{M}$ parameters).

### Q3: What is Weight Tying and why is it beneficial?
**Answer**:
Weight tying forces the target input embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ and the final output projection matrix $\mathbf{W}_{proj}^T \in \mathbb{R}^{V \times d}$ to share the exact same memory weights. This eliminates $16-30\text{M}$ redundant parameters, regularizes the model, and guarantees that input and output semantic representations are perfectly aligned.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Decoder has 3 sublayers: Masked Self-Attention, Cross-Attention, FFN.
- Output linear projection maps $d_{model} \to V$ (vocabulary size).
- Weight tying reuses embedding matrix for final projection.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=DI2_hrAulYo)
- [Lecture Video](https://www.youtube.com/watch?v=DI2_hrAulYo)
- [Official Course Notes](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)

---
