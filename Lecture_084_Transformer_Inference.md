# Lecture 084: Transformer Inference | How Inference is Done Step by Step

> **CampusX 100 Days of Deep Learning** | Video ID: `FtsMOzlwxws` | Duration: 48m 15s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=FtsMOzlwxws) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
This capstone lecture explores how a trained Transformer is executed during inference to generate real-world text.

While training is fully parallel (Teacher Forcing across all target tokens simultaneously), **inference is fundamentally autoregressive and sequential**!

Step-by-step inference pipeline:

1. Pass input sentence through the **Encoder once**; cache $\mathbf{K}_{enc}, \mathbf{V}_{enc}$.

2. Initialize decoder with `<SOS>` (Start of Sequence) token.

3. Decoder generates probabilities over vocabulary; pick next token.

4. Append selected token to decoder input and repeat step 3.

5. Terminate when `<EOS>` (End of Sequence) token is emitted or maximum length is reached.

Decoding Strategies compared:

- **Greedy Search:** Chooses $\arg\max$; fast but myopic and repetitive.

- **Beam Search:** Maintains top-$B$ most probable candidate sequences; high quality for translation.

- **Top-K & Top-p (Nucleus) Sampling:** Stochastic sampling methods that power creative generation in modern LLMs.


## 2. Key Definitions & Formal Terminology
- **Autoregressive Inference**: The sequential inference process where a model generates one token at a time, appending each newly generated token to its input for the next time step.
- **KV Caching**: An inference optimization where Key and Value projections from past time steps are cached in GPU memory, avoiding redundant $\mathcal{O}(N^2)$ recomputation at each generation step.
- **Beam Search**: A heuristic search algorithm that explores a graph by expanding the top-$B$ (beam width) most promising partial sequence candidates at each step.
- **Top-p (Nucleus) Sampling**: Sampling from the smallest set of top tokens whose cumulative probability exceeds threshold $p$ (e.g., $p=0.9$), dynamically expanding or contracting candidate pool based on model confidence.


## 3. Mathematical Formulations & Derivations
**Decoding Algorithms Formulation:**



1. **Greedy Search:**

   $$\hat{y}_t = \arg\max_{w \in V} P(w \mid y_{<t}, \mathbf{X})$$



2. **Beam Search ($B$ candidates):**

   Maintains set $\mathcal{B}_t$ of $B$ sequences maximizing cumulative log-probability:

   $$\text{Score}(y_1, \dots, y_t) = \sum_{i=1}^t \log P(y_i \mid y_{<i}, \mathbf{X})$$

   Length normalized score (to prevent bias against longer translations):

   $$\text{Score}_{norm} = \frac{1}{t^\alpha} \sum_{i=1}^t \log P(y_i \mid y_{<i}, \mathbf{X}) \quad (\alpha \approx 0.7)$$



3. **Top-p (Nucleus) Sampling Pool $\mathcal{V}^{(p)}$:**

   Smallest subset of tokens such that:

   $$\sum_{w \in \mathcal{V}^{(p)}} P(w \mid y_{<t}) \ge p$$

   Re-normalize probabilities across $\mathcal{V}^{(p)}$ and sample multinomial.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Transformer Autoregressive Inference Loop:**

```

Input: "The cat sat" ---> [ ENCODER ] ===> Cached K_enc, V_enc

                                                |

Step 1: Decoder Input: [<SOS>]             ---> [ DECODER ] ---> Predicts: "Le"

                                                |

Step 2: Decoder Input: [<SOS>, "Le"]       ---> [ DECODER ] ---> Predicts: "chat"

                                                |

Step 3: Decoder Input: [<SOS>, "Le","chat"]---> [ DECODER ] ---> Predicts: "s'est"

                                                |

Step 4: Decoder Input: [ ... ]             ---> [ DECODER ] ---> Predicts: [<EOS>] (Halt!)

```


## 5. Implementation Code Snippet
```python

import numpy as np



# KV Caching & Autoregressive Inference Loop Schema

def transformer_generate(encoder, decoder, input_tokens, max_len=50, sos_token=1, eos_token=2):

    # 1. Run encoder ONCE

    enc_output = encoder(input_tokens)

    

    # 2. Initialize target sequence with <SOS>

    target_seq = [sos_token]

    

    for step in range(max_len):

        dec_input = np.array([target_seq])

        # 3. Decoder forward pass

        logits = decoder(dec_input, enc_output)[:, -1, :] # Logits of latest token

        

        # 4. Greedy pick or Top-p sample

        next_token = int(np.argmax(logits, axis=-1)[0])

        target_seq.append(next_token)

        

        # 5. Check termination

        if next_token == eos_token:

            break

            

    return target_seq

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Transformer training fully parallel ($\mathcal{O}(1)$ steps) while inference is sequential ($\mathcal{O}(T)$ steps)?
**Answer**:
During training, the ground truth target sentence is already known. We feed all target tokens simultaneously and use **Causal Masking** to prevent look-ahead, evaluating loss across all tokens in one parallel pass. During inference, future tokens do not exist; each token must be generated, sampled, and fed back into the input before the next token can be predicted.

### Q2: What is KV Caching and why is it mandatory in modern LLM serving?
**Answer**:
At step $t$, the decoder re-evaluates self-attention for all preceding tokens $1, \dots, t-1$. Computing $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ for past tokens repeatedly at every step wastes massive compute ($\mathcal{O}(T^2)$). **KV Caching** stores the Key and Value tensors of past tokens in GPU memory; at step $t$, the model only computes $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ for the newest single token and appends $\mathbf{k}_t, \mathbf{v}_t$ to the cache, slashing inference latency to $\mathcal{O}(T)$.

### Q3: Compare Greedy Search, Beam Search, and Nucleus (Top-p) Sampling.
**Answer**:
**Greedy Search** selects the single highest-probability token at each step; fast but gets stuck in repetitive loops. **Beam Search** explores the top-$B$ globally most probable sequence paths; produces precise, formal translations but sounds robotic for creative writing. **Nucleus (Top-p) Sampling** samples dynamically from the top cumulative $p$ probability mass, introducing human-like linguistic variety while preventing gibberish outliers.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Inference is autoregressive (sequential) even though training is parallel.
- KV Caching stores past keys and values to prevent $\mathcal{O}(T^2)$ recomputation.
- Beam Search is best for translation; Top-p / Top-K sampling is best for open-ended generation.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=FtsMOzlwxws)
- [Lecture Video](https://www.youtube.com/watch?v=FtsMOzlwxws)
- [Official Course Notes](c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
