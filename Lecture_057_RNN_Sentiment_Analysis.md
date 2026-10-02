# Lecture 057: RNN Sentiment Analysis | End-to-End Keras Code Example

> **CampusX 100 Days of Deep Learning** | Video ID: `JgnbwKnHMZQ` | Duration: 39m 10s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=JgnbwKnHMZQ) | **Transcript Status**: Available (en-US, 5188 words)

---

## 1. Executive Summary & Core Intuition
This lecture builds an end-to-end sentiment classification pipeline on the IMDb dataset (50,000 movie reviews) using Keras and SimpleRNN.

The essential NLP preprocessing and modeling pipeline:

1. **Tokenization:** Mapping raw text strings into discrete integer token indices (`Tokenizer`).

2. **Vocabulary Limiting:** Restricting to the top $V$ most frequent words (e.g., $V = 10,000$) and mapping rare words to `<OOV>` (Out-of-Vocabulary).

3. **Padding / Truncating:** Using `pad_sequences` to ensure uniform sequence length $T$ across all reviews (padding shorter reviews with zeros, truncating longer reviews).

4. **Embedding Layer:** Learning dense continuous vector representations ($\mathbf{E} \in \mathbb{R}^{V \times D}$) that capture semantic similarity (replacing high-dimensional, sparse one-hot vectors).

5. **Many-to-One Classification:** Feeding the embedded word vectors through `SimpleRNN(32)`, extracting the final state $\mathbf{h}_T$, and predicting binary sentiment with `Dense(1, activation='sigmoid')`.


## 2. Key Definitions & Formal Terminology
- **Embedding Layer**: A learnable lookup table $\mathbf{E} \in \mathbb{R}^{V \times D}$ that maps discrete integer word tokens into continuous dense semantic vectors of dimension $D$ (e.g., $D=128$).
- **Sequence Padding**: Adding special padding tokens (typically zeros) to the beginning (`pre`) or end (`post`) of variable-length sequences to assemble uniform rectangular mini-batch tensors.
- **Many-to-One RNN**: An RNN topology that ingests an entire multi-step sequence $(x_1, \dots, x_T)$ and emits a single classification decision at the conclusion of the sequence.


## 3. Mathematical Formulations & Derivations
**Embedding Layer Mathematical Operation:**

Let vocabulary size be $V$, embedding dimension be $D$.

The embedding weight matrix is $\mathbf{E} \in \mathbb{R}^{V \times D}$.

For a word with one-hot vector $\mathbf{w}_i \in \{0, 1\}^V$:

$$\mathbf{e}_i = \mathbf{w}_i^T \mathbf{E} = \mathbf{E}[i, :] \in \mathbb{R}^{1 \times D}$$

The embedding layer performs a simple direct row-indexing lookup without matrix multiplication, saving massive memory!



**Parameter Count in Sentiment Architecture:**

1. **Embedding Layer:** $V \times D = 10,000 \times 32 = \mathbf{320,000}$.

2. **SimpleRNN(32):** $(D \times H) + (H \times H) + H = (32 \times 32) + (32 \times 32) + 32 = 1024 + 1024 + 32 = \mathbf{2,080}$.

3. **Dense(1):** $(H \times 1) + 1 = 32 + 1 = \mathbf{33}$.

Total Parameters: $322,113$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**IMDb Sentiment Analysis Pipeline:**

```

Raw Review: "This movie was absolutely wonderful"

      |

[ Tokenizer ] --------> [ 12, 19, 14, 450, 280 ] (Integer IDs)

      |

[ pad_sequences ] ----> Fixed length 200: [ 0, 0, ..., 12, 19, 14, 450, 280 ]

      |

[ Embedding(10000, 32) ] -> Dense vectors: (Batch, 200, 32)

      |

[ SimpleRNN(32, return_sequences=False) ] -> Final state h_T: (Batch, 32)

      |

[ Dense(1, Sigmoid) ] -> Sentiment Probability: p in (0, 1) (Positive / Negative)

```


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow.keras.preprocessing.sequence import pad_sequences



# 1. Load IMDb Data

vocab_size = 10000

max_len = 200

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)



# 2. Pad sequences to uniform length

X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')

X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')



# 3. Model Architecture

model = models.Sequential([

    layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),

    layers.SimpleRNN(32, return_sequences=False), # Many-to-One

    layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Pre-padding (`padding='pre'`) generally superior to Post-padding (`padding='post'`) in Vanilla RNNs?
**Answer**:
In Many-to-One RNNs, only the final state $\mathbf{h}_T$ is passed to the classifier. If you post-pad with zeros, the final time steps are meaningless zero tokens, causing the hidden state to decay and forget the real words seen earlier. Pre-padding places zeros at the beginning, so the final time steps contain real words, leaving the memory fresh.

### Q2: Why is an Embedding layer preferred over One-Hot Encoding?
**Answer**:
One-hot vectors are high-dimensional ($10,000+$), orthogonal (cosine distance between any two words is 0, failing to capture semantic similarity), and sparse. Dense embeddings compress words into low-dimensional continuous vectors ($64-300D$) where semantically related words ('king' and 'queen') cluster close together.

### Q3: What limitation did you observe when training SimpleRNN on reviews with `max_len=500`?
**Answer**:
Accuracy plateaus around 80-84%, and training suffers from severe vanishing gradients. The network completely forgets opinions expressed at the start of a 500-word review. Solving this requires LSTMs.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Use pre-padding (`padding='pre'`) for Many-to-One RNNs.
- Embedding layer maps token IDs to dense vectors: shape `(batch, max_len, embed_dim)`.
- SimpleRNN struggles on sequences longer than 50-100 steps.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=JgnbwKnHMZQ)
- [Lecture Video](https://www.youtube.com/watch?v=JgnbwKnHMZQ)
- [Colab Notebook](https://colab.research.google.com/drive/1uY7NEHi59w4FkB8TViwLjUDKxgCA8W5G?usp=sharing)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
