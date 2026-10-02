# Lecture 063: LSTM Part 3 | Next Word Predictor & Text Generation Project

> **CampusX 100 Days of Deep Learning** | Video ID: `fiqo6uPCJVI` | Duration: 44m 30s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=fiqo6uPCJVI) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
How do Large Language Models generate text?

This lecture builds a functional character/word-level Next Word Predictor and generative model using LSTMs in Keras.

The generative training loop:

1. Ingest a corpus of text (e.g., Shakespeare or Sherlock Holmes).

2. Generate rolling $N$-gram input sequences: for sequence "Deep learning is fascinating", inputs are:

   - "Deep" $\to$ "learning"

   - "Deep learning" $\to$ "is"

   - "Deep learning is" $\to$ "fascinating"

3. Pad all sequences to uniform length.

4. Train an LSTM with Softmax output over the vocabulary.

5. **Autoregressive Inference:** To generate text, feed a seed phrase, predict the probability distribution over the next word, sample a token, append it to the seed, and repeat iteratively!


## 2. Key Definitions & Formal Terminology
- **Language Modeling**: The task of estimating the joint probability distribution of sequences of words: $P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$.
- **Autoregressive Generation**: A generative process where the model predicts the next token conditioned on previously generated tokens, feeding its own output back as input for the next step.
- **Temperature Sampling**: A hyperparameter $T$ that scales logits before Softmax to control generation creativity ($T < 1.0$ makes text conservative/repetitive; $T > 1.0$ makes text creative/chaotic).


## 3. Mathematical Formulations & Derivations
**Language Model Factorization via Chain Rule of Probability:**

$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_{<t})$$



**Temperature-Scaled Softmax Sampling Formulation:**

Given raw output logit vector $\mathbf{z}$ and temperature hyperparameter $\tau > 0$:



$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^V \exp(z_j / \tau)}$$



- As $\tau \to 0$: Distribution collapses into a one-hot Argmax (Greedy Decoding / Zero Temperature).

- As $\tau = 1.0$: Standard Softmax probabilities.

- As $\tau \to \infty$: Distribution flattens into a uniform random distribution (Maximum Entropy / Chaos).


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Autoregressive Generation Loop Flowchart:**

```

Seed: "The neural network"

        |

  [ LSTM Model ]

        |

Logits ---> [ Temperature Softmax (T=0.7) ] ---> Sampled Word: "learns"

        |

Append to Seed: "The neural network learns"

        |

  [ Repeat Loop ]

```


## 5. Implementation Code Snippet
```python

import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, models



# Temperature sampling function

def sample_with_temperature(logits, temperature=0.7):

    logits = np.asarray(logits).astype('float64')

    logits = logits / temperature

    exp_logits = np.exp(logits - np.max(logits))

    probs = exp_logits / np.sum(exp_logits)

    # Multinomial sampling

    return np.random.choice(len(probs), p=probs)



# Text generation loop

def generate_text(model, tokenizer, seed_text, num_words, max_seq_len, temp=0.8):

    output_text = seed_text

    for _ in range(num_words):

        token_list = tokenizer.texts_to_sequences([output_text])[0]

        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')

        predicted_logits = model.predict(token_list, verbose=0)[0]

        next_index = sample_with_temperature(predicted_logits, temperature=temp)

        output_word = tokenizer.index_word.get(next_index, "")

        output_text += " " + output_word

    return output_text

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why does Greedy Search ($\arg\max$) often produce repetitive and degenerate text?
**Answer**:
Greedy search picks the locally most probable token at each step. In language, common high-probability words ('the', 'of', 'and') dominate, causing the model to quickly enter infinite loops (e.g., 'the model of the model of the model'). Sampling with moderate temperature ($0.7-0.8$) introduces stochastic variety that matches natural human language distributions.

### Q2: How does an $N$-gram sequence generator format training data?
**Answer**:
For a sentence of $K$ tokens, it generates $K-1$ training pairs: `(token[0], token[1])`, `(token[0:2], token[2])`, ..., `(token[0:K-1], token[K-1])`. All input prefixes are zero-padded to `max_len - 1` and target labels are categorical integers.

### Q3: What is Perplexity in language modeling?
**Answer**:
Perplexity is the standard evaluation metric for language models, defined as the exponentiated cross-entropy loss: $\text{PPL} = \exp(\mathcal{L}_{CE}) = \exp\left(-\frac{1}{N}\sum \ln P(w_i \mid w_{<i})\right)$. A lower perplexity indicates the model is less surprised by real test text.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Autoregressive generation feeds predicted tokens back as next inputs.
- Temperature scales logits: low $T$ = conservative, high $T$ = creative.
- Perplexity $\text{PPL} = \exp(\mathcal{L}_{CE})$ measures language model quality.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=fiqo6uPCJVI)
- [Lecture Video](https://www.youtube.com/watch?v=fiqo6uPCJVI)
- [Colab Notebook](https://colab.research.google.com/drive/1e55Lnl0I0gFgzrbOwEAGsqmnKKwAWpRO?usp=sharing)

---
