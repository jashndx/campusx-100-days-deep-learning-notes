# Lecture 080: Transformer Architecture Part 1 | Complete Encoder Deep Dive

> **CampusX 100 Days of Deep Learning** | Video ID: `Vs87qcdm8l0` | Duration: 42m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=Vs87qcdm8l0) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
This lecture synthesizes all previous concepts into the complete, rigorous architectural blueprint of the **Transformer Encoder**.

The Encoder is a stack of $N=6$ identical layers.

Each Encoder layer contains two fundamental sub-layers:

1. **Sub-layer 1:** Multi-Head Self-Attention Mechanism.

2. **Sub-layer 2:** Position-wise Feed-Forward Network (FFN).

Around each of these two sub-layers, a **Residual Skip Connection** is employed, followed by **Layer Normalization**:

$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

The Feed-Forward Network expands the representation from $d_{model}=512$ up to $d_{ff}=2048$ with a ReLU/GELU activation, and then projects it back down to $512$, acting as a distributed key-value memory store that processes each token independently in parallel.


## 2. Key Definitions & Formal Terminology
- **Transformer Encoder**: The sub-network of the Transformer responsible for reading and encoding the input sequence into a stack of rich continuous representations.
- **Position-wise Feed-Forward Network (FFN)**: A two-layer fully connected network applied identically and separately to each position: $\\text{FFN}(x) = \\max(0, x\\mathbf{W}_1 + \\mathbf{b}_1)\\mathbf{W}_2 + \\mathbf{b}_2$.
- **Add & Norm Block**: The combination of a residual identity skip connection followed by layer normalization.
- **Expansion Factor**: The ratio of intermediate FFN dimension to model dimension (standard: $d_{ff} / d_{model} = 2048 / 512 = 4$).


## 3. Mathematical Formulations & Derivations
**Complete Mathematical Step-by-Step of One Encoder Layer:**



Given input tensor from previous layer $\mathbf{H}^{[l-1]} \in \mathbb{R}^{N \times d_{model}}$:



**Step 1: Multi-Head Self-Attention:**

$$\mathbf{H}_{att} = \text{MultiHead}(\mathbf{H}^{[l-1]}, \mathbf{H}^{[l-1]}, \mathbf{H}^{[l-1]})$$



**Step 2: Residual Addition & Layer Normalization (Sub-layer 1):**

$$\mathbf{H}^{(1)} = \text{LayerNorm}\left(\mathbf{H}^{[l-1]} + \mathbf{H}_{att}\right)$$



**Step 3: Position-wise Feed-Forward Network:**

$$\mathbf{H}_{ffn} = \max\left(0, \mathbf{H}^{(1)}\mathbf{W}_1 + \mathbf{b}_1\right)\mathbf{W}_2 + \mathbf{b}_2$$

Where $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ (with $d_{model}=512, d_{ff}=2048$).



**Step 4: Residual Addition & Layer Normalization (Sub-layer 2):**

$$\mathbf{H}^{[l]} = \text{LayerNorm}\left(\mathbf{H}^{(1)} + \mathbf{H}_{ffn}\right)$$



**Parameter Accounting for 1 Encoder Layer ($d=512, d_{ff}=2048$):**

- MHA: $4 \times d^2 + 4d = 4(512^2) + 4(512) = 1,050,624$

- LayerNorm 1: $2 \times d = 1,024$

- FFN: $(d \times d_{ff} + d_{ff}) + (d_{ff} \times d + d) = 2(512 \times 2048) + 2048 + 512 = 2,097,152 + 2,560 = 2,099,712$

- LayerNorm 2: $2 \times d = 1,024$

- **Total per Encoder Layer:** $\approx \mathbf{3,152,384 \text{ Parameters}} \approx \mathbf{3.15 \text{ Million}}$.

- Across $N=6$ stacked layers: $6 \times 3.15\text{M} \approx \mathbf{18.9 \text{ Million Parameters}}$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Single Encoder Layer Architecture Block:**

```

Input: H^[l-1] -----------------------------------------------\

       |                                                      | (Residual)

  [ Multi-Head Attention ]                                    |

       |                                                      |

       +<-----------------------------------------------------/

       |

  [ LayerNorm ] ---> H^(1) -----------------------------------\

       |                                                      | (Residual)

  [ Feed-Forward Network: Dense(2048, ReLU) -> Dense(512) ]   |

       |                                                      |

       +<-----------------------------------------------------/

       |

  [ LayerNorm ] ---> Output: H^[l]

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers



class TransformerEncoderLayer(layers.Layer):

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout_rate=0.1):

        super().__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)

        self.ffn = tf.keras.Sequential([

            layers.Dense(d_ff, activation='relu'),

            layers.Dense(d_model)

        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = layers.Dropout(dropout_rate)

        self.dropout2 = layers.Dropout(dropout_rate)



    def call(self, x, training=False, mask=None):

        # Sub-layer 1: MHA + Add & Norm

        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask)

        x = self.layernorm1(x + self.dropout1(attn_output, training=training))

        

        # Sub-layer 2: FFN + Add & Norm

        ffn_output = self.ffn(x)

        x = self.layernorm2(x + self.dropout2(ffn_output, training=training))

        return x

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does the Position-wise FFN project dimension up by $4\times$ ($512 \to 2048$) before projecting back down?
**Answer**:
Geva et al. (2021) demonstrated that the FFN functions as a key-value associative memory. Expanding to $2048$ creates a high-dimensional sparse intermediate space where thousands of factual concepts and linguistic patterns can be stored and triggered by the non-linear activation before being compressed back into the model dimension.

### Q2: What is the role of the two Residual connections in each Encoder layer?
**Answer**:
Residual connections preserve identity gradient highways ($1 + \frac{\partial F}{\partial x}$). Without residual connections, deep stacks of $6$ to $24$ Transformer layers would suffer severe gradient vanishing and representation collapse, preventing effective backpropagation.

### Q3: Why is it called a 'Position-wise' Feed-Forward Network?
**Answer**:
Because the exact same two-layer MLP is applied to every token position independently and identically: $\text{FFN}(\mathbf{x}_i) = \max(0, \mathbf{x}_i \mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$. There is zero interaction between different token positions inside the FFN (all cross-token interaction occurs exclusively in the Multi-Head Attention layer).

## 7. Crucial Exam Takeaways & Common Pitfalls
- Encoder layer = MHA (Sublayer 1) + FFN (Sublayer 2), each wrapped in Add & Norm.
- FFN expands $4\times$ ($d_{model} \to 4d_{model} \to d_{model}$).
- One base encoder layer contains $\approx 3.15$ million parameters.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=Vs87qcdm8l0)
- [Lecture Video](https://www.youtube.com/watch?v=Vs87qcdm8l0)
- [Official Course Notes](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
