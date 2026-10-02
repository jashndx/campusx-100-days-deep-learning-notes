# Lecture 068: Encoder Decoder | Sequence-to-Sequence (Seq2Seq) Architecture

> **CampusX 100 Days of Deep Learning** | Video ID: `KiL74WsgxoA` | Duration: 38m 20s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=KiL74WsgxoA) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
How do you map an input sequence of length $T_x$ to an output sequence of a completely different length $T_y$ (e.g., translating a 10-word English sentence into a 6-word German sentence)?

Sutskever et al. and Cho et al. (2014) invented the **Encoder-Decoder (Seq2Seq)** architecture.

The workflow:

1. **The Encoder:** An RNN (or LSTM) ingests the input tokens $\mathbf{x}_1, \dots, \mathbf{x}_{T_x}$ one by one, updating its hidden state. At the final time step $T_x$, its final hidden state is extracted as the **Context Vector $\mathbf{c}$**.

2. **The Bottleneck:** The context vector $\mathbf{c}$ is supposed to represent a complete, condensed numerical summary of the *entire* input sentence.

3. **The Decoder:** Another RNN initializes its hidden state with $\mathbf{c}$ and generates the target tokens $\mathbf{y}_1, \dots, \mathbf{y}_{T_y}$ autoregressively, starting with `<SOS>` (Start of Sequence) until emitting `<EOS>` (End of Sequence).

The fatal flaw of classic Seq2Seq: **The Information Bottleneck**. Forcing an entire 50-word paragraph into a single fixed-size 512D vector causes catastrophic information loss!


## 2. Key Definitions & Formal Terminology
- **Sequence-to-Sequence (Seq2Seq)**: A deep learning framework that transforms an input sequence of one domain/length into an output sequence of another domain/length using coupled encoder and decoder sub-networks.
- **Context Vector ($\mathbf{c}$)**: The fixed-length numerical vector produced by the encoder that encapsulates the semantic information of the input sequence.
- **Information Bottleneck**: The theoretical compression bottleneck in Seq2Seq where all nuanced syntactic and semantic details of long input sequences are compressed into a single fixed-size vector.
- **Teacher Forcing**: A training strategy for recurrent decoders where the true ground truth token from the previous time step $y_{t-1}^*$ is supplied as input rather than the model's own (potentially incorrect) prediction $\hat{y}_{t-1}$.


## 3. Mathematical Formulations & Derivations
**Mathematical Seq2Seq Formulations:**



1. **Encoder Pass:**

   $$\mathbf{h}_t^e = f_e(\mathbf{h}_{t-1}^e, \mathbf{x}_t), \quad t \in [1, T_x]$$

   $$\mathbf{c} = q(\mathbf{h}_1^e, \dots, \mathbf{h}_{T_x}^e) = \mathbf{h}_{T_x}^e \quad (\text{Final state})$$



2. **Decoder Initialization & Autoregressive Step:**

   $$\mathbf{s}_0 = \mathbf{c}$$

   $$\mathbf{s}_t = f_d(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c})$$

   $$\hat{y}_t = \text{Softmax}(\mathbf{W}_y \mathbf{s}_t + \mathbf{b}_y)$$



**Teacher Forcing Loss Function:**

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T_y} \log P(y_t^* \mid y_{<t}^*, \mathbf{X})$$

During training, conditioned on true targets $y_{<t}^*$; during inference, conditioned on generated predictions $\hat{y}_{<t}$.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Seq2Seq Information Flow:**

```

ENCODER:                                         DECODER:

x_1 -> [LSTM] -> h_1                               s_0 = c 

x_2 -> [LSTM] -> h_2                                 |

x_3 -> [LSTM] -> h_3 = Context Vector c ======>  [LSTM] -> s_1 -> Predict y_1 ("Le")

                                                     |

                                                 [LSTM] -> s_2 -> Predict y_2 ("chat")

                                                     |

                                                 [LSTM] -> s_3 -> Predict <EOS>

```

Notice how ALL information from $x_1, x_2, x_3$ must squeeze through the tiny bridge $\mathbf{c}$.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Basic Seq2Seq Architecture in Keras

latent_dim = 256



# 1. Encoder

encoder_inputs = layers.Input(shape=(None, 100), name='encoder_inputs')

encoder = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')

encoder_outputs, state_h, state_c = encoder(encoder_inputs)

encoder_states = [state_h, state_c] # The Context Vector!



# 2. Decoder

decoder_inputs = layers.Input(shape=(None, 100), name='decoder_inputs')

decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')

# Initialize decoder state with encoder's final state

decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

decoder_dense = layers.Dense(10000, activation='softmax', name='decoder_output')

decoder_outputs = decoder_dense(decoder_outputs)



seq2seq_model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: What is the 'Information Bottleneck' problem in basic Seq2Seq?
**Answer**:
The encoder is forced to compress sentences of arbitrary length (10, 50, or 100 words) into a single, fixed-size vector $\mathbf{c}$ (e.g., 512 floats). As sentence length increases beyond 15-20 words, BLEU score and translation quality drop catastrophically because early sentence information is lost. This bottleneck motivated the invention of the **Attention Mechanism**.

### Q2: What is Teacher Forcing and what is 'Exposure Bias'?
**Answer**:
**Teacher Forcing:** Feeding the ground truth token $y_{t-1}^*$ as decoder input during training, which speeds up convergence. **Exposure Bias:** During inference, ground truth is absent, so the decoder feeds its own predictions $\hat{y}_{t-1}$. If it makes a single mistake, errors cascade down the sequence. Solutions include Scheduled Sampling.

### Q3: Why did Sutskever et al. reverse the source sentence in their 2014 paper?
**Answer**:
Reversing the input sentence (feeding $x_T, x_{T-1}, \dots, x_1$) placed the first source word $x_1$ right next to the first target word $y_1$ in the unrolled graph. This drastically shortened the minimal time lag between corresponding words, providing immediate gradient flow and boosting BLEU scores by over 5 points.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Encoder compresses sequence into fixed Context Vector $\mathbf{c} = \mathbf{h}_{T_x}$.
- Information bottleneck causes severe performance collapse on sequences $> 20$ words.
- Teacher forcing accelerates training, but introduces exposure bias.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=KiL74WsgxoA)
- [Lecture Video](https://www.youtube.com/watch?v=KiL74WsgxoA)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
