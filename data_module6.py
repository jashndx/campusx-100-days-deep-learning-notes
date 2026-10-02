"""
Module 6: Sequence-to-Sequence, Attention Mechanisms & Complete Transformer Architecture (Lectures 68 - 84)
"""

MODULE_6_LECTURES = [
    {
        "index": 68,
        "title": "Encoder Decoder | Sequence-to-Sequence (Seq2Seq) Architecture",
        "video_id": "KiL74WsgxoA",
        "duration_str": "38m 20s",
        "core_intuition": r"""
How do you map an input sequence of length $T_x$ to an output sequence of a completely different length $T_y$ (e.g., translating a 10-word English sentence into a 6-word German sentence)?
Sutskever et al. and Cho et al. (2014) invented the **Encoder-Decoder (Seq2Seq)** architecture.
The workflow:
1. **The Encoder:** An RNN (or LSTM) ingests the input tokens $\mathbf{x}_1, \dots, \mathbf{x}_{T_x}$ one by one, updating its hidden state. At the final time step $T_x$, its final hidden state is extracted as the **Context Vector $\mathbf{c}$**.
2. **The Bottleneck:** The context vector $\mathbf{c}$ is supposed to represent a complete, condensed numerical summary of the *entire* input sentence.
3. **The Decoder:** Another RNN initializes its hidden state with $\mathbf{c}$ and generates the target tokens $\mathbf{y}_1, \dots, \mathbf{y}_{T_y}$ autoregressively, starting with `<SOS>` (Start of Sequence) until emitting `<EOS>` (End of Sequence).
The fatal flaw of classic Seq2Seq: **The Information Bottleneck**. Forcing an entire 50-word paragraph into a single fixed-size 512D vector causes catastrophic information loss!
""",
        "key_definitions": [
            ("Sequence-to-Sequence (Seq2Seq)", "A deep learning framework that transforms an input sequence of one domain/length into an output sequence of another domain/length using coupled encoder and decoder sub-networks."),
            (r"Context Vector ($\mathbf{c}$)", "The fixed-length numerical vector produced by the encoder that encapsulates the semantic information of the input sequence."),
            ("Information Bottleneck", "The theoretical compression bottleneck in Seq2Seq where all nuanced syntactic and semantic details of long input sequences are compressed into a single fixed-size vector."),
            ("Teacher Forcing", r"A training strategy for recurrent decoders where the true ground truth token from the previous time step $y_{t-1}^*$ is supplied as input rather than the model's own (potentially incorrect) prediction $\hat{y}_{t-1}$.")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            ("What is the 'Information Bottleneck' problem in basic Seq2Seq?", 
             r"The encoder is forced to compress sentences of arbitrary length (10, 50, or 100 words) into a single, fixed-size vector $\mathbf{c}$ (e.g., 512 floats). As sentence length increases beyond 15-20 words, BLEU score and translation quality drop catastrophically because early sentence information is lost. This bottleneck motivated the invention of the **Attention Mechanism**."),
            ("What is Teacher Forcing and what is 'Exposure Bias'?", 
             r"**Teacher Forcing:** Feeding the ground truth token $y_{t-1}^*$ as decoder input during training, which speeds up convergence. **Exposure Bias:** During inference, ground truth is absent, so the decoder feeds its own predictions $\hat{y}_{t-1}$. If it makes a single mistake, errors cascade down the sequence. Solutions include Scheduled Sampling."),
            ("Why did Sutskever et al. reverse the source sentence in their 2014 paper?", 
             r"Reversing the input sentence (feeding $x_T, x_{T-1}, \dots, x_1$) placed the first source word $x_1$ right next to the first target word $y_1$ in the unrolled graph. This drastically shortened the minimal time lag between corresponding words, providing immediate gradient flow and boosting BLEU scores by over 5 points.")
        ],
        "exam_takeaways": [
            r"Encoder compresses sequence into fixed Context Vector $\mathbf{c} = \mathbf{h}_{T_x}$.",
            "Information bottleneck causes severe performance collapse on sequences $> 20$ words.",
            "Teacher forcing accelerates training, but introduces exposure bias."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=KiL74WsgxoA")
        ]
    },
    {
        "index": 69,
        "title": "Attention Mechanism | Bahdanau Additive Attention in Depth",
        "video_id": "rj5V6q6-XUM",
        "duration_str": "45m 15s",
        "core_intuition": r"""
How do you break the Seq2Seq information bottleneck?
Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio (2014/2015) published the revolutionary paper: "Neural Machine Translation by Jointly Learning to Align and Translate".
The Core Intuition: **Stop compressing the entire sentence into a single static vector!**
Instead of discarding intermediate encoder hidden states, the decoder keeps *all* encoder hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_{T_x})$.
When generating output word $t$, the decoder looks at its current state $\mathbf{s}_{t-1}$ and compares it with every encoder state $\mathbf{h}_i$ to compute **Attention Weights $\alpha_{ti}$** (a probability distribution summing to 1).
A dynamic, custom **Context Vector $\mathbf{c}_t$** is constructed as a weighted average: $\mathbf{c}_t = \sum \alpha_{ti} \mathbf{h}_i$.
When generating the word "cat", the network pays 95% attention to the encoder state for "chat" and 5% to other words!
""",
        "key_definitions": [
            ("Attention Mechanism", "A mechanism that allows a decoder to dynamically focus on specific relevant parts of the input sequence when generating each output token."),
            ("Alignment Score ($e_{ti}$)", "A scalar quantifying how well the inputs around position $i$ match the output at position $t$."),
            (r"Attention Weights ($\alpha_{ti}$)", "Softmax-normalized alignment scores representing the probability distribution over source tokens for decoding step $t$."),
            (r"Dynamic Context Vector ($\mathbf{c}_t$)", r"The weighted linear combination of all encoder hidden states: $\\mathbf{c}_t = \\sum_{i=1}^{T_x} \\alpha_{ti} \\mathbf{h}_i$.")
        ],
        "mathematical_formulations": r"""
**The Complete Bahdanau (Additive) Attention Equations:**

Let encoder hidden states be $\mathbf{h}_1, \dots, \mathbf{h}_{T_x} \in \mathbb{R}^h$, and current decoder state be $\mathbf{s}_{t-1} \in \mathbb{R}^s$.

1. **Alignment Model (Additive / MLP Score):**
   $$e_{ti} = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$$
   Where $\mathbf{W}_a \in \mathbb{R}^{a \times s}, \mathbf{U}_a \in \mathbb{R}^{a \times h}, \mathbf{v}_a \in \mathbb{R}^{a \times 1}$ are learnable attention parameters.

2. **Softmax Normalization (Attention Weights):**
   $$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^{T_x} \exp(e_{tk})}, \quad \sum_{i=1}^{T_x} \alpha_{ti} = 1$$

3. **Dynamic Context Vector Computation:**
   $$\mathbf{c}_t = \sum_{i=1}^{T_x} \alpha_{ti} \mathbf{h}_i$$

4. **Decoder State & Prediction Update:**
   $$\mathbf{s}_t = f_d([\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t])$$
   $$\hat{y}_t = \text{Softmax}(\mathbf{W}_o [\mathbf{s}_t, \mathbf{c}_t])$$
""",
        "architecture_and_algorithm": r"""
**Bahdanau Attention Dynamic Flowchart:**
```
Encoder States:     h_1        h_2        h_3        h_4
                     |          |          |          |
Alignment Scores:  e_t1       e_t2       e_t3       e_t4  <--- Compared with Decoder State s_{t-1}
                     |          |          |          |
Softmax:          a_t1       a_t2       a_t3       a_t4   (Sums to 1.0)
                     \          \          /          /
Dynamic Context:                  c_t = SUM(a_ti * h_i)
                                           |
                                           v
                              Decoder Step t ---> Emits y_t
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

class BahdanauAttention(layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W = layers.Dense(units)
        self.U = layers.Dense(units)
        self.V = layers.Dense(1)

    def call(self, query, values):
        # query: decoder hidden state s_{t-1} -> shape: (batch, 1, hidden_dim)
        # values: encoder hidden states h -> shape: (batch, seq_len, hidden_dim)
        
        # score = V^T * tanh(W*query + U*values)
        score = self.V(tf.nn.tanh(self.W(query) + self.U(values)))
        
        # attention_weights shape: (batch, seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # context_vector shape: (batch, hidden_dim)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        return context_vector, attention_weights
```""",
        "exam_viva_qa": [
            ("How does Bahdanau Attention eliminate the Seq2Seq Information Bottleneck?", 
             r"Instead of compressing all input tokens into one fixed vector $\mathbf{c} = \mathbf{h}_{T_x}$, the attention mechanism retains *all* intermediate encoder representations. At every decoding step, a bespoke context vector $\mathbf{c}_t$ is synthesized on-the-fly, allowing the decoder to look directly at any token in the source sentence regardless of sentence length."),
            ("Why is Bahdanau Attention called 'Additive' Attention?", 
             r"Because inside the alignment scoring function, the decoder state and encoder state are projected and combined via vector addition: $\mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s}_{t-1} + \mathbf{U}_a \mathbf{h}_i)$. Contrast this with Luong's Multiplicative attention which uses dot products."),
            ("How do Attention Weights provide model interpretability?", 
             r"Plotting the matrix of attention weights $\alpha_{ti}$ as a 2D heatmap produces an explicit word-alignment matrix showing exactly which source words the model focused on when emitting each translated target word (e.g., showing how English adjective-noun order maps to French noun-adjective order).")
        ],
        "exam_takeaways": [
            r"Formula to memorize: $\mathbf{c}_t = \sum \alpha_{ti} \mathbf{h}_i$.",
            "Attention weights sum to 1.0 via Softmax over source sequence $T_x$.",
            "Alignment heatmap provides direct visual explainability."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=rj5V6q6-XUM")
        ]
    },
    {
        "index": 70,
        "title": "Bahdanau Attention Vs Luong Attention | Additive vs Multiplicative",
        "video_id": "0hZT4_fHfNQ",
        "duration_str": "31m 40s",
        "core_intuition": r"""
Following Bahdanau's breakthrough, Minh-Thang Luong et al. (2015) introduced key architectural refinements that simplified attention and improved computational efficiency.
This lecture contrasts the two classical attention paradigms:
1. **Bahdanau (Additive) Attention:** Uses a feedforward network (vector addition) to compute alignment scores. It calculates attention *before* updating the decoder hidden state, using $\mathbf{s}_{t-1}$.
2. **Luong (Multiplicative) Attention:** Replaces the expensive additive MLP with matrix dot products (General / Dot / Concat). It calculates attention *after* updating the decoder hidden state, using $\mathbf{s}_t$.
Multiplicative attention is mathematically faster and more memory efficient because it translates directly into highly optimized BLAS matrix multiplications on modern GPUs.
""",
        "key_definitions": [
            ("Luong (Multiplicative) Attention", r"An attention variant developed by Luong et al. (2015) that calculates alignment scores using dot products and utilizes the current decoder hidden state $\mathbf{s}_t$."),
            ("Global vs Local Attention", "Global attention attends to all source tokens; Local attention attends only to a small centered window $[p_t - D, p_t + D]$ of source tokens to save compute on long texts."),
            ("Dot-Product Alignment Score", r"The simplest alignment score: $e_{ti} = \mathbf{s}_t^T \mathbf{h}_i$, which requires no trainable parameters and measures direct vector cosine similarity.")
        ],
        "mathematical_formulations": r"""
**Comparison of Alignment Scoring Functions:**

Given decoder query $\mathbf{s}$ and encoder key $\mathbf{h}$:

1. **Bahdanau (Additive):**
   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}_a^T \tanh(\mathbf{W}_a \mathbf{s} + \mathbf{U}_a \mathbf{h})$$

2. **Luong (Dot):** *(Requires $\text{dim}(\mathbf{s}) = \text{dim}(\mathbf{h})$)*
   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{h}$$

3. **Luong (General):** *(Introduces learnable bilinear matrix $\mathbf{W}_a$)*
   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{W}_a \mathbf{h}$$

4. **Luong (Concat):**
   $$\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{v}_a^T \tanh(\mathbf{W}_a [\mathbf{s} ; \mathbf{h}])$$

**Timing Difference:**
- Bahdanau: Uses $\mathbf{s}_{t-1}$ to compute $\mathbf{c}_t$, then computes $\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1}, \mathbf{c}_t)$.
- Luong: Computes $\mathbf{s}_t = f(\mathbf{s}_{t-1}, y_{t-1})$ first, then uses $\mathbf{s}_t$ to compute $\mathbf{c}_t$, combining them into attentional vector:
  $$\widetilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t ; \mathbf{s}_t])$$
""",
        "architecture_and_algorithm": r"""
**Key Differences Summary Table:**

| Feature | Bahdanau Attention (2014) | Luong Attention (2015) |
| :--- | :--- | :--- |
| **Score Type** | Additive ($\mathbf{v}^T \tanh(\mathbf{W}\mathbf{s} + \mathbf{U}\mathbf{h})$) | Multiplicative ($\mathbf{s}^T \mathbf{W} \mathbf{h}$ or $\mathbf{s}^T \mathbf{h}$) |
| **Decoder State Used** | Previous state $\mathbf{s}_{t-1}$ | Current state $\mathbf{s}_t$ |
| **Computational Speed**| Slower (evaluates MLP) | Faster (BLAS matrix multiply) |
| **Scope** | Always Global | Global or Local |
| **Attentional Vector** | Emitted directly from RNN | Combined via $\widetilde{\mathbf{s}}_t = \tanh(\mathbf{W}_c [\mathbf{c}_t; \mathbf{s}_t])$ |
""",
        "code_snippet": r"""```python
import tensorflow as tf

# Luong Dot-Product Alignment Score
def luong_dot_score(decoder_state, encoder_states):
    # decoder_state: (batch, 1, dim)
    # encoder_states: (batch, seq_len, dim)
    # Transpose encoder states: (batch, dim, seq_len)
    # Matrix multiply computes dot product for all tokens simultaneously!
    score = tf.matmul(decoder_state, encoder_states, transpose_b=True)
    return score # shape: (batch, 1, seq_len)
```""",
        "exam_viva_qa": [
            ("Why is Multiplicative (Dot) Attention faster than Additive Attention?", 
             r"Dot-product attention can be implemented as a single, highly optimized batch matrix multiplication (`tf.matmul` or `torch.bmm`), which fully saturates GPU matrix cores. Additive attention involves multiple linear projections, an element-wise addition, and a non-linear $\tanh$ activation, which are memory-bandwidth bound."),
            ("What is the difference between Global and Local Attention in Luong et al.?", 
             r"**Global Attention:** Attends to *every* source token across the entire sentence, with computational complexity $\mathcal{O}(T_x)$ per decoding step. **Local Attention:** Predicts an aligned center position $p_t$ on the source sentence and computes attention strictly within a small window $[p_t - D, p_t + D]$, reducing computational complexity to a constant $\mathcal{O}(2D + 1)$."),
            ("Which alignment score in Luong attention requires no trainable parameters?", 
             r"The **Dot** score: $\text{score}(\mathbf{s}, \mathbf{h}) = \mathbf{s}^T \mathbf{h}$. It computes direct cosine-like vector similarity, requiring zero additional weights, provided hidden dimensions match.")
        ],
        "exam_takeaways": [
            r"Bahdanau = Additive (uses $\mathbf{s}_{t-1}$); Luong = Multiplicative (uses $\mathbf{s}_t$).",
            "Multiplicative attention scales efficiently via GPU matrix multiplication.",
            r"Luong General score: $\mathbf{s}^T \mathbf{W}_a \mathbf{h}$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=0hZT4_fHfNQ")
        ]
    },
    {
        "index": 71,
        "title": "Introduction to Transformers | The Architecture That Changed AI",
        "video_id": "BjRVS2wTtcA",
        "duration_str": "35m 10s",
        "core_intuition": r"""
In 2017, Ashish Vaswani et al. at Google Brain and Google Research published "Attention Is All You Need", introducing the **Transformer**—the architecture that completely revolutionized AI and powers modern LLMs (GPT-4, Gemini, Claude, LLaMA).
The foundational paradigm shift:
Up to 2017, attention was merely an auxiliary mechanism bolted onto recurrent (LSTM/GRU) backbones.
The Transformer proposed an audacious thesis: **Recurrence is completely unnecessary. Discard recurrence entirely. Attention is ALL you need!**
Why discard recurrence?
1. Recurrence is inherently sequential: $\mathbf{h}_t$ depends on $\mathbf{h}_{t-1}$, making GPU parallelization across time impossible.
2. In recurrent networks, information between token $1$ and token $100$ must traverse $100$ sequential hops, risking degradation.
In a Transformer, **every token connects directly to every other token in a single operational step ($\mathcal{O}(1)$ path length)** via Self-Attention!
All $T$ tokens are processed in parallel, unlocking massive GPU training scalability.
""",
        "key_definitions": [
            ("Transformer", "A deep learning architecture relying entirely on self-attention mechanisms to compute representations of its input and output without using sequence-aligned recurrent networks or convolutions."),
            ("Self-Attention (Intra-Attention)", "An attention mechanism relating different positions of a single sequence in order to compute a representation of the exact same sequence."),
            ("Sequential Computation Constraint", "The computational bottleneck of RNNs where time step $t$ cannot begin processing until the output of step $t-1$ has finished computing."),
            (r"Path Length ($\mathcal{O}(1)$)", "The number of operational forward hops required for a signal to propagate between any two arbitrary positions in an input sequence.")
        ],
        "mathematical_formulations": r"""
**Path Length and Complexity Comparison Table (Vaswani et al., 2017):**

Let $n$ be sequence length, $d$ be representation dimension, and $k$ be convolution kernel size.

| Layer Type | Complexity per Layer | Max Path Length | Sequential Operations |
| :--- | :--- | :--- | :--- |
| **Recurrent (RNN/LSTM)** | $\mathcal{O}(n \cdot d^2)$ | $\mathcal{O}(n)$ | $\mathcal{O}(n)$ (Sequential Bottleneck!) |
| **Convolutional (1D CNN)**| $\mathcal{O}(k \cdot n \cdot d^2)$ | $\mathcal{O}(\log_k(n))$ | $\mathcal{O}(1)$ (Parallel) |
| **Self-Attention** | $\mathcal{O}(n^2 \cdot d)$ | $\mathbf{\mathcal{O}(1)}$ | $\mathbf{\mathcal{O}(1)}$ (Fully Parallel!) |

Notice that for Self-Attention:
- Sequential operations: $\mathcal{O}(1)$ $\implies$ All tokens processed simultaneously across GPU cores!
- Max path length: $\mathcal{O}(1)$ $\implies$ Token 1 connects to Token 10,000 in a single direct step!
""",
        "architecture_and_algorithm": r"""
**High-Level Transformer Architectural Topology:**
```
Input Sequence  ---> [ Positional Encoding ] ---> [ N x Encoder Blocks ] ---\
                                                                             ---> Cross-Attention
Target Sequence ---> [ Positional Encoding ] ---> [ N x Decoder Blocks ] ---/
                                                         |
                                                  [ Linear Head ]
                                                         |
                                                  [ Softmax Probabilities ]
```
Both Encoder and Decoder consist of $N$ identical stacked layers (standard: $N=6$).
""",
        "code_snippet": r"""```python
import tensorflow as tf

# In a Transformer, an entire batch of sequences is passed simultaneously:
# Input shape: (Batch Size, Sequence Length, Embedding Dim)
# No temporal for-loop! The entire 3D tensor is processed in one matrix pass!
X = tf.random.normal((32, 128, 512)) # 32 sentences, 128 words, 512 dimensions
```""",
        "exam_viva_qa": [
            (r"Why does the Transformer have a maximum path length of $\mathcal{O}(1)$ compared to $\mathcal{O}(n)$ in RNNs?", 
             "In an RNN, for token 1 to communicate with token $n$, the information must sequentially pass through $n$ intermediate recurrent cell state transitions ($n$ steps). In Self-Attention, an attention weight is computed directly between every pair of positions $(i, j)$ via dot product in a single matrix operation, connecting any two tokens in exactly 1 hop."),
            (r"If Self-Attention is $\mathcal{O}(n^2 \cdot d)$, why is it faster to train than an RNN with $\mathcal{O}(n \cdot d^2)$?", 
             r"When sequence length $n$ is smaller than embedding dimension $d$ (e.g., $n=512, d=768$, typical in NLP), $n^2 \cdot d < n \cdot d^2$. More importantly, the RNN operations are *sequential* (preventing hardware parallelization), while the Transformer operations are *matrix multiplications* that fully saturate thousands of GPU tensor cores simultaneously."),
            (r"What is the major trade-off of the $\mathcal{O}(n^2)$ complexity in Transformers?", 
             r"Memory and compute scale quadratically with sequence length $n$. Doubling the context window from 2,048 to 4,096 tokens increases attention matrix memory by $4\times$. This quadratic bottleneck inspired efficient/linear attention variants (FlashAttention, Linformer).")
        ],
        "exam_takeaways": [
            "Transformers eliminate recurrence entirely in favor of parallel Self-Attention.",
            r"Sequential operations are $\mathcal{O}(1)$, unlocking massive GPU training parallelism.",
            r"Path length between any two tokens is $\mathcal{O}(1)$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=BjRVS2wTtcA")
        ]
    },
    {
        "index": 72,
        "title": "What is Self-Attention? | The Query, Key, Value Metaphor",
        "video_id": "XnGGmvpDLA0",
        "duration_str": "32m 40s",
        "core_intuition": r"""
What is Self-Attention conceptually, and how does a sentence attend to itself?
Consider the sentence: "The **animal** didn't cross the street because **it** was too tired."
What does the pronoun "**it**" refer to? The animal or the street?
To a human, "tired" implies an animate creature, so "it" refers to the animal.
Self-Attention allows the word "it" to look across all other words in the sentence, compute affinity scores, and blend the representation of "animal" into the representation of "it"!
To compute this interaction dynamically, Vaswani et al. borrowed the **Query, Key, Value (Q, K, V)** retrieval metaphor from database and search engine systems:
- **Query ($\mathbf{Q}$):** 'What am I looking for?' (The current word seeking context).
- **Key ($\mathbf{K}$):** 'What do I contain?' (Every word's identity tag).
- **Value ($\mathbf{V}$):** 'What is my actual content/meaning?' (The information delivered if a match is made).
""",
        "key_definitions": [
            (r"Query ($\mathbf{Q}$)", "A vector representing the current token's search intent when probing the sequence for contextual relationships."),
            (r"Key ($\mathbf{K}$)", "A vector representing a token's indexable label that is compared against incoming queries via dot products."),
            (r"Value ($\mathbf{V}$)", "A vector representing the actual informative content that is aggregated into the output representation once query and key match."),
            ("Contextualized Representation", "A dynamic token representation that incorporates weighted semantic information from relevant surrounding tokens.")
        ],
        "mathematical_formulations": r"""
**The Database Analogy Formalization:**
In a Python dictionary or database query:
$$\text{Output} = \text{lookup}(\text{query}) \longrightarrow \text{match}(\text{query}, \text{key}_i) \implies \text{retrieve}(\text{value}_i)$$

In classical databases, key matching is hard/binary:
$$\text{Output} = \begin{cases} \mathbf{v}_i & \text{if } \mathbf{q} == \mathbf{k}_i \\ 0 & \text{otherwise} \end{cases}$$

In Transformer Self-Attention, matching is **Soft and Differentiable**:
Every query matches *every* key to a fractional degree:
$$\text{Attention Weight } \alpha_i = \text{Softmax}(\mathbf{q} \cdot \mathbf{k}_i)$$
$$\text{Output} = \sum_{i=1}^N \alpha_i \mathbf{v}_i$$
The output is a soft, weighted sum of all values in the sequence!
""",
        "architecture_and_algorithm": r"""
**Query, Key, Value Information Pipeline:**
```
Input Word Embedding: "it" (x_i)
        |
        +---> [ W_Q ] ---> Query Vector (q_i)  --- (Dot product with all Keys) ---> Attention Weights
        |                                                                                  |
        +---> [ W_K ] ---> Key Vector (k_i)                                                v
        |                                                                          (Weighted Sum of Values)
        +---> [ W_V ] ---> Value Vector (v_i) ----------------------------------> Contextualized Vector (z_i)
```
Every token produces its own unique $\mathbf{q}, \mathbf{k}, \mathbf{v}$ via three learnable linear projection matrices.
""",
        "code_snippet": r"""```python
import numpy as np

# Conceptual Self-Attention for single word
# Words: ["The", "animal", "crossed", "it"]
# If query is "it", its dot product with key("animal") is high!
query_it = np.array([0.1, 0.9, 0.2])
key_animal = np.array([0.1, 0.8, 0.3])
key_street = np.array([0.8, 0.1, 0.1])

score_animal = np.dot(query_it, key_animal) # 0.01 + 0.72 + 0.06 = 0.79 (High match!)
score_street = np.dot(query_it, key_street) # 0.08 + 0.09 + 0.02 = 0.19 (Low match!)
```""",
        "exam_viva_qa": [
            ("Explain the Query, Key, Value metaphor using YouTube search as an analogy.", 
             "When you search for 'deep learning tutorial' in YouTube's search bar, that text is your **Query**. YouTube compares your query against the titles, tags, and metadata (**Keys**) of millions of cataloged videos. The similarity match between query and keys determines rank/weights. Finally, YouTube delivers the actual video content (**Values**) of the matching results."),
            ("Why does every word have THREE separate vectors ($Q, K, V$) instead of just using its original embedding vector?", 
             r"A word serves different roles in different interactions: when seeking context, it acts as a Query; when providing context to another word, it acts as a Key; when delivering semantic content, it acts as a Value. Having separate learnable projection matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) decouples these three roles, providing vastly greater expressive capacity."),
            (r"What does the self-attention output $\mathbf{z}_i$ for a word represent?", 
             "It represents the updated, contextualized embedding of word $i$. It retains the core identity of the original word while infusing relevant contextual nuances from all other words in the sentence that the word attended to.")
        ],
        "exam_takeaways": [
            "Query = What I am looking for; Key = What I contain; Value = What I deliver.",
            r"Soft database retrieval: $\text{Output} = \sum \text{Softmax}(Q \cdot K^T) V$.",
            "Transforms static word embeddings into contextualized representations."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=XnGGmvpDLA0")
        ]
    },
    {
        "index": 73,
        "title": "Self-Attention in Transformers | Full Mathematical Derivation & Code",
        "video_id": "-tCKPl_8Xb8",
        "duration_str": "41m 20s",
        "core_intuition": r"""
This lecture establishes the complete mathematical formulation of the Scaled Dot-Product Attention mechanism and derives its vectorized matrix implementation from scratch.
Tracing the full computational sequence for input matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$:
1. Project $\mathbf{X}$ into $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ via learned weight matrices $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$.
2. Compute raw affinity scores matrix: $\mathbf{S} = \mathbf{Q}\mathbf{K}^T$.
3. Scale by $\frac{1}{\sqrt{d_k}}$ to maintain unit variance and prevent gradient vanishing in Softmax.
4. Apply row-wise Softmax to compute the Attention Matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$.
5. Multiply attention weights by Value matrix: $\mathbf{Z} = \mathbf{A}\mathbf{V}$.
The entire multi-token, bidirectional contextualization is computed in **two matrix multiplications**!
""",
        "key_definitions": [
            ("Scaled Dot-Product Attention", r"The mathematical core of the Transformer: $\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V}$."),
            (r"Projection Matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$)", "Learnable weight matrices of shape $(d_{model}, d_k)$ used to project input embeddings into query, key, and value subspaces."),
            (r"Attention Matrix ($\mathbf{A}$)", r"A square $N \times N$ stochastic matrix where entry $A_{i, j}$ represents the attention weight that token $i$ pays to token $j$ (each row sums to 1.0).")
        ],
        "mathematical_formulations": r"""
**The Fundamental Transformer Equation:**

$$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q \quad (\mathbf{Q} \in \mathbb{R}^{N \times d_k})$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}_K \quad (\mathbf{K} \in \mathbb{R}^{N \times d_k})$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}_V \quad (\mathbf{V} \in \mathbb{R}^{N \times d_v})$$

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$

**Dimensionality Dimensional Analysis:**
- Input sequence: $\mathbf{X} \in \mathbb{R}^{N \times d_{model}}$ ($N$ tokens, $d_{model}=512$)
- Weights: $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d_{model} \times d_k}$, $\mathbf{W}_V \in \mathbb{R}^{d_{model} \times d_v}$
- Query $\times$ Key Transpose:
  $$\mathbf{Q}\mathbf{K}^T = (N \times d_k) \times (d_k \times N) = \mathbf{(N \times N)}$$
- Softmax over rows:
  $$\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times N}$$
- Attention $\times$ Value:
  $$\mathbf{Z} = \mathbf{A}\mathbf{V} = (N \times N) \times (N \times d_v) = \mathbf{(N \times d_v)}$$
Output retains the exact same sequence length $N$ as the input!
""",
        "architecture_and_algorithm": r"""
**Step-by-Step Computational Graph:**
```
Input X (N x d) ---> [ W_Q, W_K, W_V ] ---> Q (N x d_k), K (N x d_k), V (N x d_v)
                                                     |         |
                                                     v         v
                                              Matrix Mul: Q * K^T (N x N)
                                                           |
                                                           v
                                              Scale: / sqrt(d_k)
                                                           |
                                                           v
                                              Softmax (row-wise) ---> Attention Matrix A (N x N)
                                                                            |
                                                                            v
                                                                 Matrix Mul: A * V (N x d_v)
                                                                            |
                                                                            v
                                                               Output Contextualized Z (N x d_v)
```
""",
        "code_snippet": r"""```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    # 1. Matmul Q and K^T
    scores = np.matmul(Q, K.swapaxes(-2, -1))
    # 2. Scale by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)
    # 3. Optional Masking (for decoder causal attention)
    if mask is not None:
        scaled_scores += (mask * -1e9)
    # 4. Softmax over last axis
    exp_scores = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    # 5. Multiply by V
    output = np.matmul(attention_weights, V)
    return output, attention_weights
```""",
        "exam_viva_qa": [
            (r"Calculate the shape of the attention matrix $\mathbf{A}$ for a sequence of 50 words with embedding size 512.", 
             r"The attention matrix $\mathbf{A} = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)$ has shape $(N \times N) = \mathbf{50 \times 50}$. Every entry $A_{ij}$ indicates how much attention word $i$ pays to word $j$."),
            ("Why must Softmax be applied across rows (`axis=-1`) rather than columns?", 
             r"Row $i$ of the matrix corresponds to the Query of token $i$ looking across all Keys $j \in \{1, \dots, N\}$. The attention distribution for token $i$ must form a valid probability distribution over all tokens $j$ ($\sum_{j=1}^N A_{ij} = 1.0$), which requires normalizing across rows."),
            ("How does Self-Attention handle sentences of different lengths in a batch?", 
             r"Padding tokens are masked out before Softmax. A large negative number ($-10^9$ or $-\infty$) is added to the scores of padding positions. When exponentiated ($e^{-\infty} = 0$), the attention weights for pad tokens become exactly zero, ensuring real tokens ignore padding.")
        ],
        "exam_takeaways": [
            r"Formula to memorize: $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$.",
            r"Attention matrix shape is $(N \times N)$, independent of embedding dimension $d$.",
            r"Output shape $(N \times d_v)$ matches input sequence length."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=-tCKPl_8Xb8")
        ]
    },
    {
        "index": 74,
        "title": "Scaled Dot Product Attention | Why Do We Scale by Sqrt(d_k)?",
        "video_id": "r7mAt0iVqwo",
        "duration_str": "25m 30s",
        "core_intuition": r"""
Why did the authors of the Transformer divide the dot products by the square root of key dimension $\sqrt{d_k}$?
Why not use standard unscaled dot products $\mathbf{Q}\mathbf{K}^T$?
This lecture provides the mathematical proof:
When the dimension $d_k$ is large (e.g., $d_k = 64$ or $512$), computing the dot product between two independent random vectors involves summing $d_k$ individual products: $\sum_{i=1}^{d_k} q_i k_i$.
By the central limit theorem, the variance of this sum grows linearly with dimension: $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$.
For $d_k = 64$, standard deviation is $\sqrt{64} = 8$!
Values in the dot-product matrix blow up into large magnitudes ($+20, -25$).
Passing large values into the Softmax function pushes outputs into extreme saturated regions where the function is virtually flat!
As a result, the derivative of Softmax vanishes to zero ($\approx 0$), halting backpropagation gradient flow!
Dividing by $\sqrt{d_k}$ normalizes the variance strictly back to $1.0$, keeping gradients alive.
""",
        "key_definitions": [
            (r"Scaling Factor ($\frac{1}{\sqrt{d_k}}$)", "The normalization factor applied to dot products in self-attention to preserve unit variance regardless of vector dimension."),
            ("Softmax Saturation", "A pathology where large input logit magnitudes force one Softmax probability to $1.0$ and all others to $0.0$, driving derivatives to near-zero and freezing gradient flow."),
            ("Variance of Dot Product", "The mathematical property showing that the sum of $d$ independent product terms with zero mean and unit variance has a total variance of $d$.")
        ],
        "mathematical_formulations": r"""
**Mathematical Proof that $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k$:**

Assume components $q_i$ and $k_i$ are independent random variables with zero mean and unit variance:
$$\mathbb{E}[q_i] = 0, \quad \text{Var}(q_i) = 1$$
$$\mathbb{E}[k_i] = 0, \quad \text{Var}(k_i) = 1$$

Let scalar dot product be $Z = \mathbf{q} \cdot \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$.
1. **Expected Value:**
   $$\mathbb{E}[Z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i k_i] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

2. **Variance of Product of Independent Variables:**
   $$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2] \mathbb{E}[k_i^2] - 0 = (1)(1) = 1$$

3. **Variance of the Sum of $d_k$ Independent Variables:**
   $$\text{Var}(Z) = \text{Var}\left( \sum_{i=1}^{d_k} q_i k_i \right) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = \sum_{i=1}^{d_k} 1 = \mathbf{d_k}$$
   Standard deviation: $\sigma = \sqrt{d_k}$.

**Applying the Scaling Factor:**
If we scale $Z$ by $\frac{1}{\sqrt{d_k}}$:
$$\text{Var}\left( \frac{Z}{\sqrt{d_k}} \right) = \left(\frac{1}{\sqrt{d_k}}\right)^2 \text{Var}(Z) = \frac{1}{d_k} \cdot d_k = \mathbf{1.0}$$
The variance is normalized back to $1.0$, completely independent of dimension $d_k$!
""",
        "architecture_and_algorithm": r"""
**Softmax Gradient Vanishing Dynamics:**
Recall Softmax derivative: $\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)$.
- If inputs are unscaled ($z_1 = 30, z_2 = -20$):
  $$\sigma_1 \approx 1.0, \quad \sigma_2 \approx 0.0$$
  $$\text{Derivative: } \sigma_1(1 - \sigma_1) \approx 1.0(1 - 1.0) = \mathbf{0.0} \implies \text{Gradients Vanish!}$$
- With scaling ($z_1 = 3.0, z_2 = -2.0$):
  $$\sigma_1 \approx 0.99, \sigma_2 \approx 0.01 \implies \text{Healthy gradient flow!}$$
""",
        "code_snippet": r"""```python
import numpy as np

# Numerical simulation proving variance growth
d_k = 100
N_trials = 10000

q = np.random.randn(N_trials, d_k) # Mean 0, Var 1
k = np.random.randn(N_trials, d_k) # Mean 0, Var 1

dot_products = np.sum(q * k, axis=1)
print(f"Unscaled Variance (expected {d_k}): {np.var(dot_products):.2f}")

scaled_dot_products = dot_products / np.sqrt(d_k)
print(f"Scaled Variance (expected 1.0): {np.var(scaled_dot_products):.2f}")
# Output verifies: Unscaled Var ~ 100.0, Scaled Var ~ 1.0!
```""",
        "exam_viva_qa": [
            ("Derive why the variance of the dot product of two $d$-dimensional random vectors is $d$.", 
             r"Let $Z = \sum_{i=1}^d q_i k_i$. For independent zero-mean unit-variance variables, $\text{Var}(q_i k_i) = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \times 1 = 1$. Since the variance of the sum of independent random variables is the sum of their variances: $\text{Var}(Z) = \sum_{i=1}^d 1 = d$."),
            ("Why does high variance in logits cause gradients to vanish in Softmax?", 
             r"When logits have large variance, the exponentiated values differ by orders of magnitude. The largest logit dominates the denominator, driving its Softmax probability $\sigma_k \to 1.0$ and all other probabilities $\sigma_j \to 0$. The derivative of Softmax is $\sigma_i(\delta_{ij} - \sigma_j)$, which evaluates to $1(1-1) = 0$ for the winner and $0(0) = 0$ for losers, driving all gradients to zero."),
            ("Does Dot-Product Attention outperform Additive Attention for small $d_k$?", 
             "For small dimensions $d_k$, additive attention and unscaled dot-product attention perform similarly. For large dimensions $d_k$, unscaled dot-product attention degrades significantly due to vanishing gradients, while scaled dot-product attention performs equally well and is significantly faster.")
        ],
        "exam_takeaways": [
            r"Be able to reproduce the proof: $\\text{Var}(\\sum q_i k_i) = d_k$.",
            r"Dividing by $\\sqrt{d_k}$ normalizes variance to $1.0$.",
            "Prevents Softmax saturation and vanishing gradients during backprop."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=r7mAt0iVqwo")
        ]
    },
    {
        "index": 75,
        "title": "Self-Attention Geometric Intuition | Subspaces & Projections",
        "video_id": "5ZgGuujZSbs",
        "duration_str": "28m 40s",
        "core_intuition": r"""
This lecture provides the visual and geometric intuition behind Self-Attention in high-dimensional vector spaces.
Geometrically:
1. Input word vectors exist as points/arrows in a $d$-dimensional semantic hyperspace (e.g., 512 dimensions).
2. The dot product $\mathbf{q} \cdot \mathbf{k}$ measures the **geometric cosine similarity** scaled by vector lengths:
   $$\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos(\theta)$$
   If two word concepts are semantically aligned in the subspace, their angle $\theta$ is small, $\cos(\theta) \to 1$, and their dot product is large.
3. The linear projection matrices $\mathbf{W}_Q, \mathbf{W}_K$ act as **subspace coordinate transforms**—they rotate and project raw embeddings into dedicated query and key hyperplanes where relevant task relationships become co-linear.
4. Multiplying by $\mathbf{V}$ performs a **convex combination (barycentric interpolation)**: the new token vector is a center-of-mass coordinate pulled toward the values of the words it attended to!
""",
        "key_definitions": [
            ("Cosine Similarity", r"A measure of directional similarity between two non-zero vectors in an inner product space: $\\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$."),
            ("Convex Combination", r"A linear combination of points where all coefficients are non-negative and sum strictly to $1$: $\\sum \\alpha_i \\mathbf{v}_i$ where $\\alpha_i \\ge 0, \\sum \\alpha_i = 1$."),
            ("Semantic Subspace", "A lower-dimensional vector space spanned by learned projection matrices where specific linguistic relationships (e.g., subject-verb agreement, gender) are isolated.")
        ],
        "mathematical_formulations": r"""
**Geometric Interpretation of Attention as a Barycentric Hull:**

Because $\sum_{j=1}^N A_{ij} = 1.0$ and $A_{ij} \ge 0$ for all $j$:
$$\mathbf{z}_i = \sum_{j=1}^N A_{ij} \mathbf{v}_j$$

Geometrically, the output vector $\mathbf{z}_i$ is strictly constrained to lie inside the **Convex Hull** formed by the set of all Value vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N\}$:

$$\mathbf{z}_i \in \text{Conv}(\{\mathbf{v}_1, \dots, \mathbf{v}_N\})$$

If word $i$ pays 100% attention to word $j$, $\mathbf{z}_i$ jumps to vertex $\mathbf{v}_j$. 
If it balances attention between two words equally, $\mathbf{z}_i$ settles exactly at their midpoint!
""",
        "architecture_and_algorithm": r"""
**Geometric Vector Movement in Semantic Space:**
```
                          Value("animal")
                              *
                             /
                            /  <-- z_("it") is pulled heavily towards "animal"!
                           * z_("it")
                          /
                         /
      Value("street") *
```
The representation of "it" physically shifts its spatial coordinates toward "animal" in the embedding space!
""",
        "code_snippet": r"""```python
import numpy as np

# Geometric interpolation demonstration
v_animal = np.array([2.0, 5.0])
v_street = np.array([8.0, 1.0])

# Attention weights: 80% animal, 20% street
alpha_animal = 0.8
alpha_street = 0.2

# New contextualized vector
z_it = alpha_animal * v_animal + alpha_street * v_street
print("Contextualized vector:", z_it) # [3.2, 4.2] -> Lies on line segment!
```""",
        "exam_viva_qa": [
            ("Why is the dot product used as a similarity measure in self-attention rather than Euclidean distance?", 
             r"Dot product combines both angle (directional alignment) and magnitude (importance/salience). Furthermore, computing dot products across an entire sequence is a single matrix multiplication $\mathbf{Q}\mathbf{K}^T$, which is orders of magnitude faster on GPUs than computing pairwise Euclidean distances $\|q_i - k_j\|_2$."),
            (r"What does it mean geometrically that $\mathbf{z}_i$ lies in the convex hull of the Value vectors?", 
             "It means self-attention cannot synthesize entirely new vector magnitudes or directions out of thin air; it can only re-weight, interpolate, and blend existing informational values that were already present in the sequence."),
            (r"How do learned projection matrices $\mathbf{W}_Q, \mathbf{W}_K$ enable attention to identify non-obvious relationships?", 
             r"Two words might have orthogonal or distant raw embeddings. The linear projection matrices $\mathbf{W}_Q$ and $\mathbf{W}_K$ project those vectors into a transformed latent subspace where their projected coordinates become aligned (high dot product), allowing the model to learn task-specific contextual dependencies.")
        ],
        "exam_takeaways": [
            r"Dot product measures directional similarity: $\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$.",
            r"Output $\mathbf{z}_i$ is a convex combination lying inside the convex hull of Value vectors.",
            "Learned matrices project tokens into specialized functional subspaces."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=5ZgGuujZSbs")
        ]
    },
    {
        "index": 76,
        "title": "Why is Self Attention Called 'Self'? | Self vs Cross Attention",
        "video_id": "o4ZVA0TuDRg",
        "duration_str": "26m 15s",
        "core_intuition": r"""
Why is this mechanism explicitly termed **Self-Attention**?
This lecture clarifies the critical distinction between **Self-Attention** and **Cross-Attention** (traditional Bahdanau/Luong attention):
- **Cross-Attention (Inter-Attention):** Connects two *different* sequences. Queries come from Sequence B (Decoder target sentence), while Keys and Values come from Sequence A (Encoder source sentence).
- **Self-Attention (Intra-Attention):** Queries, Keys, and Values ALL come from the **exact same sequence**!
In Self-Attention, a sequence looks at *itself*: words inside the sentence attend to other words inside the *same* sentence to resolve pronouns, syntactic dependencies, and polysemy.
""",
        "key_definitions": [
            ("Self-Attention (Intra-Attention)", r"An attention mechanism where Queries, Keys, and Values are all derived from the same sequence: $\\mathbf{Q}, \\mathbf{K}, \\mathbf{V} = \\mathbf{X}\\mathbf{W}_Q, \\mathbf{X}\\mathbf{W}_K, \\mathbf{X}\\mathbf{W}_V$."),
            ("Cross-Attention", "An attention mechanism where Queries originate from one sequence (e.g., decoder state), and Keys and Values originate from another sequence (e.g., encoder output)."),
            ("Intra-Sentence Dependency", "Syntactic and semantic relationships that exist strictly within the boundaries of a single sentence (e.g., subject-verb agreement, antecedent resolution).")
        ],
        "mathematical_formulations": r"""
**Mathematical Comparison of Origin Sources:**

1. **Self-Attention (Encoder or Decoder Self-Attention):**
   $$\mathbf{X} \in \mathbb{R}^{N \times d}$$
   $$\mathbf{Q} = \mathbf{X}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}_V$$
   *All three tensors originate from the single input $\mathbf{X}$!*

2. **Cross-Attention (Decoder-Encoder Attention):**
   Let $\mathbf{X}_{dec} \in \mathbb{R}^{M \times d}$ (Decoder sequence) and $\mathbf{H}_{enc} \in \mathbb{R}^{N \times d}$ (Encoder output):
   $$\mathbf{Q} = \mathbf{X}_{dec} \mathbf{W}_Q \quad (\text{From Decoder!})$$
   $$\mathbf{K} = \mathbf{H}_{enc} \mathbf{W}_K \quad (\text{From Encoder!})$$
   $$\mathbf{V} = \mathbf{H}_{enc} \mathbf{W}_V \quad (\text{From Encoder!})$$
   Queries probe across into the external encoder memory!
""",
        "architecture_and_algorithm": r"""
**Comparison Matrix:**

| Feature | Self-Attention | Cross-Attention |
| :--- | :--- | :--- |
| **Input Source(s)** | Single sequence $\mathbf{X}$ | Two distinct sequences ($\mathbf{X}_{target}, \mathbf{X}_{source}$) |
| **Where it occurs in Transformer** | Encoder Layers & Decoder Layer 1 | Decoder Layer 2 (Encoder-Decoder block) |
| **Core Objective** | Contextualize words within same sequence | Align target words with source words |
| **Attention Matrix Shape** | $(N \times N)$ square | $(M \times N)$ rectangular ($M = T_{dec}, N = T_{enc}$) |
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

# In Keras MultiHeadAttention:
mha = layers.MultiHeadAttention(num_heads=8, key_dim=64)

# 1. Self-Attention: query=x, value=x, key=x (defaults to value if key not given)
self_att_out = mha(query=x, value=x, key=x)

# 2. Cross-Attention: query=decoder_state, value=encoder_state, key=encoder_state
cross_att_out = mha(query=decoder_tokens, value=encoder_memory, key=encoder_memory)
```""",
        "exam_viva_qa": [
            ("Where in the complete Transformer architecture does Self-Attention occur, and where does Cross-Attention occur?", 
             "**Self-Attention** occurs in two places: (1) Inside every Encoder layer (unmasked bidirectional self-attention), and (2) Inside the first sublayer of every Decoder layer (masked causal self-attention). **Cross-Attention** occurs in the second sublayer of every Decoder layer, where queries from the decoder attend to keys and values from the encoder."),
            (r"Why is the attention matrix in Cross-Attention rectangular $(M \times N)$ rather than square $(N \times N)$?", 
             r"Because in Cross-Attention, the decoder sequence has length $M$ (target sentence length) and the encoder sequence has length $N$ (source sentence length). The dot product $\mathbf{Q}\mathbf{K}^T$ multiplies $(M \times d_k) \times (d_k \times N)$, producing an $(M \times N)$ matrix."),
            ("Can Self-Attention be applied to non-text modalities?", 
             "Yes! In Vision Transformers (ViT), image patches are treated as tokens that attend to other image patches within the *same* image (Self-Attention). In audio, temporal frames attend to other frames within the *same* audio waveform.")
        ],
        "exam_takeaways": [
            r"Self-Attention: Q, K, V all come from the same sequence $\mathbf{X}$.",
            "Cross-Attention: Q comes from Decoder; K, V come from Encoder.",
            r"Cross-attention matrix shape is rectangular: $(T_{dec} \times T_{enc})$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=o4ZVA0TuDRg")
        ]
    },
    {
        "index": 77,
        "title": "Multi-Head Attention in Transformers | Multi-Head vs Self Attention",
        "video_id": "bX2QwpjsmuA",
        "duration_str": "36m 45s",
        "core_intuition": r"""
While single-head self-attention is powerful, it has a serious limitation: average pooling.
If a word attends to multiple relationships simultaneously (e.g., in "The bank was robbed on Tuesday", the word "bank" needs to attend to "robbed" for semantic meaning, and "was" for syntactic tense), a single attention head is forced to average these different relationships together, diluting nuanced linguistic signals.
**Multi-Head Attention (MHA)** solves this by running $h$ independent self-attention mechanisms (heads) in parallel!
Each head projects $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ into a lower-dimensional subspace ($d_k = d_{model} / h$):
- Head 1 can focus on grammatical subject-verb agreement.
- Head 2 can focus on coreference resolution (pronouns).
- Head 3 can focus on direct object relationships.
The outputs of all $h$ heads are concatenated and multiplied by a final projection matrix $\mathbf{W}_O$, synthesizing multi-faceted contextual representations at **zero additional computational cost** compared to a full-rank single head!
""",
        "key_definitions": [
            ("Multi-Head Attention (MHA)", "An attention module that linearly projects queries, keys, and values $h$ times with different learned projections, computes scaled dot-product attention in parallel, concatenates the results, and projects again."),
            ("Head ($h$)", "One of the parallel attention subspaces (standard: $h=8$ heads in base Transformer, $h=16$ in large models)."),
            ("Head Dimension ($d_k$)", r"The dimensionality of each individual head: $d_k = d_v = d_{model} / h$ (for $d_{model}=512$ and $h=8 \implies d_k = 64$)."),
            (r"Output Projection Matrix ($\mathbf{W}_O$)", r"A learnable matrix of shape $(h \cdot d_v, d_{model})$ that linearly combines the concatenated multi-head outputs back into the model dimension.")
        ],
        "mathematical_formulations": r"""
**Complete Multi-Head Attention Formulations (Vaswani et al., 2017):**

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}_O$$

Where each individual head is:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \ \mathbf{K}\mathbf{W}_i^K, \ \mathbf{V}\mathbf{W}_i^V)$$

**Projection Matrix Dimensions:**
- $\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $\mathbf{W}_O \in \mathbb{R}^{(h \cdot d_v) \times d_{model}}$

**Parameter Calculation:**
For $h$ heads, each with $d_k = d_{model} / h$:
- Total weights for $Q, K, V$: $3 \times (h \times d_{model} \times d_k) = 3 \times (d_{model} \times d_{model}) = \mathbf{3 d_{model}^2}$.
- Output projection $\mathbf{W}_O$: $(h \cdot d_v) \times d_{model} = d_{model} \times d_{model} = \mathbf{d_{model}^2}$.
- **Total Parameters:** $4 d_{model}^2$ (+ biases).
*For $d_{model} = 512$:*
$$\text{Params}_{MHA} = 4 \times 512^2 = 4 \times 262,144 = \mathbf{1,048,576 \text{ Parameters}} \approx \mathbf{1.05 \text{ Million}}.$$
""",
        "architecture_and_algorithm": r"""
**Multi-Head Attention Parallel Execution Flowchart:**
```
Input (N x d_model)
       |
  +----+----+----+----+----+----+----+----+ (Split into h=8 parallel heads)
  |    |    |    |    |    |    |    |
Head1 Head2 Head3 Head4 Head5 Head6 Head7 Head8  (Each computes Attention in d_k=64)
  |    |    |    |    |    |    |    |
  +----+----+----+----+----+----+----+----+
       |
  [ Concatenate: (N x 8*64) = (N x 512) ]
       |
  [ Linear Projection W_O: (512 x 512) ]
       |
  Output (N x d_model)
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

# MultiHeadAttention in Keras
mha_layer = layers.MultiHeadAttention(
    num_heads=8,     # h = 8
    key_dim=64       # d_k = 64 (Total model dim = 8 * 64 = 512)
)

# Input tensor shape: (batch, seq_len, 512)
x = tf.random.normal((32, 50, 512))
out = mha_layer(query=x, value=x, key=x)
print("MHA Output shape:", out.shape) # (32, 50, 512)
```""",
        "exam_viva_qa": [
            ("Why does Multi-Head Attention NOT increase computational complexity compared to a full-rank single-head attention?", 
             r"Because the representation dimension is divided across the heads: $d_k = d_{model} / h$. Computing $h$ dot products of size $d_k$ requires $h \times (N^2 \cdot d_k) = N^2 \cdot (h \cdot d_k) = N^2 \cdot d_{model}$ operations—the exact same computational FLOP count as a single head operating on the full dimension $d_{model}$!"),
            ("What is the primary representational advantage of Multi-Head Attention over Single-Head Attention?", 
             "Single-head attention forces the network to average attention over multiple conflicting linguistic relationships. Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions simultaneously (e.g., syntactic structure in head 1, coreference in head 2, semantic theme in head 3)."),
            ("Calculate the total trainable parameters in a Multi-Head Attention layer with $d_{model}=768$ and $h=12$ (BERT-Base specifications).", 
             r"Using formula $\text{Params} \approx 4 \times d_{model}^2$: $4 \times (768)^2 = 4 \times 589,824 = \mathbf{2,359,296}$ weights (plus $4 \times 768 = 3,072$ biases).")
        ],
        "exam_takeaways": [
            "Formula to memorize: $d_k = d_{model} / h$.",
            "MHA parameter count is $4 d_{model}^2$.",
            "FLOP complexity of $h$ heads at dimension $d/h$ is identical to 1 head at dimension $d$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=bX2QwpjsmuA"),
            ("Colab Notebook", "https://colab.research.google.com/drive/1hXIQ77A4TYS4y3UthWF-Ci7V7vVUoxmQ")
        ]
    },
    {
        "index": 78,
        "title": "Positional Encoding in Transformers | Sinusoidal Formulations",
        "video_id": "GeoQBNNqIbM",
        "duration_str": "34m 15s",
        "core_intuition": r"""
Self-Attention is fundamentally **Permutation-Invariant**!
If you randomly scramble the word order of a sentence, the self-attention formula $\text{Softmax}(QK^T)V$ computes the exact same values, just row-permuted!
To a raw Transformer, "Dog bites man" and "Man bites dog" appear 100% mathematically identical!
In RNNs, word order was naturally baked into the sequential step-by-step recurrence.
Because Transformers process all tokens in parallel, order must be explicitly injected.
Vaswani et al. solved this by adding **Positional Encodings ($\mathbf{PE}$)** directly to the input word embeddings before the first layer:
$$\mathbf{X}_{input} = \mathbf{Embedding}(x) + \mathbf{PE}$$
Instead of learning static lookup tables, they designed an ingenious **Sinusoidal Positional Encoding** using sine and cosine waves of varying frequencies, allowing the network to extrapolate to arbitrary sequence lengths and easily learn relative positions!
""",
        "key_definitions": [
            (r"Positional Encoding ($\mathbf{PE}$)", "A tensor added to token embeddings that injects information about the absolute and relative position of tokens in a sequence."),
            ("Permutation Invariance", r"A mathematical property where a function's output does not depend on the ordering of its input elements: $f(\pi(\mathbf{X})) = \pi(f(\mathbf{X}))$. Self-attention is permutation equivariant without positional encodings."),
            ("Sinusoidal Positional Encoding", "Deterministic positional encodings constructed using sine and cosine functions across geometric frequency progressions."),
            ("Relative Position Shift Property", "The mathematical property where $PE_{pos + k}$ can be expressed as a linear transformation of $PE_{pos}$, allowing the model to easily learn relative distances.")
        ],
        "mathematical_formulations": r"""
**The Sinusoidal Positional Encoding Formulas (Vaswani et al., 2017):**

For token position $pos \in [0, N-1]$ and dimension index $i \in [0, \frac{d_{model}}{2} - 1]$:

$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i / d_{model}}} \right)$$
$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i / d_{model}}} \right)$$

Where wavelengths form a geometric progression from $2\pi$ to $10,000 \cdot 2\pi$.

**The Linear Transformation Property for Relative Positions:**
For any fixed offset $k$, there exists a linear transformation matrix $\mathbf{M}_k \in \mathbb{R}^{2 \times 2}$ such that:
$$\begin{bmatrix} PE_{(pos+k, 2i)} \\ PE_{(pos+k, 2i+1)} \end{bmatrix} = \begin{bmatrix} \cos(\omega_i k) & \sin(\omega_i k) \\ -\sin(\omega_i k) & \cos(\omega_i k) \end{bmatrix} \begin{bmatrix} PE_{(pos, 2i)} \\ PE_{(pos, 2i+1)} \end{bmatrix}$$
Where $\omega_i = \frac{1}{10000^{2i/d}}$. 
This rotational property allows self-attention to attend to relative positions ($pos + k$) purely via linear operations!
""",
        "architecture_and_algorithm": r"""
**Visualizing Positional Encoding Addition:**
```
Input Tokens:      "The"               "cat"               "sat"
                     |                   |                   |
Embeddings:      E("The")            E("cat")            E("sat")      (d_model = 512)
                     +                   +                   +
Pos Encodings:    PE(pos=0)           PE(pos=1)           PE(pos=2)    (Sinusoidal waves)
                     |                   |                   |
Combined Input:  X_0 (512-dim)       X_1 (512-dim)       X_2 (512-dim)
```
""",
        "code_snippet": r"""```python
import numpy as np
import tensorflow as tf

def get_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis] # (seq_len, 1)
    
    # Division term: 10000^(2i / d_model)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term) # Even indices: sin
    pe[:, 1::2] = np.cos(position * div_term) # Odd indices: cos
    
    return tf.constant(pe, dtype=tf.float32)
```""",
        "exam_viva_qa": [
            ("Why is Positional Encoding ADDED to word embeddings rather than CONCATENATED?", 
             "Concatenating would increase the input dimension (e.g., from 512 to $512 + k$), increasing parameters across all subsequent weight matrices throughout the entire network. Vaswani et al. showed that because the embedding space has 512 dimensions, the semantic embedding information and positional wave signals occupy nearly orthogonal subspaces, allowing addition without corrupting semantic meaning."),
            ("Why did the authors use sinusoidal functions instead of simple learned embeddings or normalized integers ($pos / N$)?", 
             "Normalizing integers by $pos / N$ fails because the scale changes depending on sentence length $N$. Learned positional embeddings cannot extrapolate to sequence lengths longer than those seen during training. Sinusoidal encodings have a fixed periodic structure that allows the model to extrapolate to unseen sequence lengths while providing the relative position linear shift property."),
            ("Why are different frequencies used across dimension $i$?", 
             "Similar to the binary representation of integers where the least significant bit alternates rapidly ($0, 1, 0, 1$) and higher bits alternate slowly ($00, 11, 00$), the low dimensions of PE have high frequencies (fine-grained position changes) and high dimensions have low frequencies (coarse-grained position changes), giving every position a unique continuous fingerprint.")
        ],
        "exam_takeaways": [
            "Self-attention without positional encoding is 100% permutation invariant.",
            r"Even dimensions use $\sin$, odd dimensions use $\cos$.",
            "Linear rotational property allows model to attend to relative distances ($pos + k$)."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=GeoQBNNqIbM"),
            ("Official Course Notes (Lectures 78-84)", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 79,
        "title": "Layer Normalization in Transformers | LayerNorm vs BatchNorm",
        "video_id": "qti0QPdaelg",
        "duration_str": "29m 30s",
        "core_intuition": r"""
Why do Transformers use **Layer Normalization (LayerNorm)** instead of **Batch Normalization (BatchNorm)**?
In Computer Vision (CNNs), Batch Normalization computes statistics across the mini-batch dimension: for a feature channel, it computes the mean across all $M$ images in the batch.
In Natural Language Processing (Transformers), this fails for two major reasons:
1. **Variable Sequence Lengths:** Different sentences have different lengths. Computing a batch mean across time step $t=150$ when only 2 sentences in a 32-sample batch are that long produces extremely noisy, unstable statistics.
2. **Dependency on Batch Size:** BatchNorm breaks when batch size is small ($B=1$ during inference).
**Layer Normalization (Ba, Kiros, Hinton, 2016)** normalizes **across the feature dimensions for each individual token independently**!
It requires no running statistics, is completely independent of batch size, works identically during training and testing, and provides rock-solid stability for Transformer training.
""",
        "key_definitions": [
            ("Layer Normalization (LayerNorm)", "A normalization technique that normalizes inputs across features within a single data instance, rather than across instances in a batch."),
            ("Pre-LN vs Post-LN", r"Architectural placement of LayerNorm: Post-LN places normalization after the residual addition ($x = \text{LN}(x + \text{Sublayer}(x))$); Pre-LN places it before ($\text{Sublayer}(\text{LN}(x)) + x$), which drastically improves gradient stability in deep models."),
            ("Feature Dimension Normalization", r"Normalizing across the $d_{model}$ vector of a single token at a single time step: $\mu = \frac{1}{d} \sum_{i=1}^d x_i$.")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            ("Why does Layer Normalization behave identically during training and testing, unlike Batch Normalization?", 
             "BatchNorm computes statistics across batch partners during training, but must switch to frozen exponential moving averages during testing. LayerNorm computes mean and variance strictly across the features of the current token within the current sample. It has zero dependency on batch partners, making training and inference mathematically identical."),
            ("Why is Pre-LN preferred over Post-LN in modern Large Language Models (like GPT-3, LLaMA)?", 
             r"In original Vaswani (Post-LN), gradients passing through residual connections must pass through LayerNorm, causing gradient magnitudes to grow or shrink unpredictably near the output, requiring a delicate learning rate warm-up. In Pre-LN, the residual shortcut ($x + \text{Sublayer}(\text{LN}(x))$) forms an uninterrupted identity highway where gradients flow unscaled directly from output to input, enabling training of models with hundreds of layers without warm-up."),
            ("What are the trainable parameters of a LayerNormalization layer in a 512-dim Transformer?", 
             r"Two vectors of shape $(512,)$: the learnable scale parameter $\boldsymbol{\gamma}$ (initialized to 1) and the shift parameter $\boldsymbol{\beta}$ (initialized to 0). Total parameters = $512 + 512 = 1,024$ parameters.")
        ],
        "exam_takeaways": [
            "BatchNorm normalizes across batch; LayerNorm normalizes across feature dimension $D$.",
            "LayerNorm has zero dependency on batch size and works identically at inference.",
            "Pre-LN stabilizes gradient flow compared to original Post-LN."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=qti0QPdaelg"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 80,
        "title": "Transformer Architecture Part 1 | Complete Encoder Deep Dive",
        "video_id": "Vs87qcdm8l0",
        "duration_str": "42m 30s",
        "core_intuition": r"""
This lecture synthesizes all previous concepts into the complete, rigorous architectural blueprint of the **Transformer Encoder**.
The Encoder is a stack of $N=6$ identical layers.
Each Encoder layer contains two fundamental sub-layers:
1. **Sub-layer 1:** Multi-Head Self-Attention Mechanism.
2. **Sub-layer 2:** Position-wise Feed-Forward Network (FFN).
Around each of these two sub-layers, a **Residual Skip Connection** is employed, followed by **Layer Normalization**:
$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$
The Feed-Forward Network expands the representation from $d_{model}=512$ up to $d_{ff}=2048$ with a ReLU/GELU activation, and then projects it back down to $512$, acting as a distributed key-value memory store that processes each token independently in parallel.
""",
        "key_definitions": [
            ("Transformer Encoder", "The sub-network of the Transformer responsible for reading and encoding the input sequence into a stack of rich continuous representations."),
            ("Position-wise Feed-Forward Network (FFN)", r"A two-layer fully connected network applied identically and separately to each position: $\\text{FFN}(x) = \\max(0, x\\mathbf{W}_1 + \\mathbf{b}_1)\\mathbf{W}_2 + \\mathbf{b}_2$."),
            ("Add & Norm Block", "The combination of a residual identity skip connection followed by layer normalization."),
            ("Expansion Factor", "The ratio of intermediate FFN dimension to model dimension (standard: $d_{ff} / d_{model} = 2048 / 512 = 4$).")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            (r"Why does the Position-wise FFN project dimension up by $4\times$ ($512 \to 2048$) before projecting back down?", 
             "Geva et al. (2021) demonstrated that the FFN functions as a key-value associative memory. Expanding to $2048$ creates a high-dimensional sparse intermediate space where thousands of factual concepts and linguistic patterns can be stored and triggered by the non-linear activation before being compressed back into the model dimension."),
            ("What is the role of the two Residual connections in each Encoder layer?", 
             r"Residual connections preserve identity gradient highways ($1 + \frac{\partial F}{\partial x}$). Without residual connections, deep stacks of $6$ to $24$ Transformer layers would suffer severe gradient vanishing and representation collapse, preventing effective backpropagation."),
            ("Why is it called a 'Position-wise' Feed-Forward Network?", 
             r"Because the exact same two-layer MLP is applied to every token position independently and identically: $\text{FFN}(\mathbf{x}_i) = \max(0, \mathbf{x}_i \mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$. There is zero interaction between different token positions inside the FFN (all cross-token interaction occurs exclusively in the Multi-Head Attention layer).")
        ],
        "exam_takeaways": [
            "Encoder layer = MHA (Sublayer 1) + FFN (Sublayer 2), each wrapped in Add & Norm.",
            r"FFN expands $4\times$ ($d_{model} \to 4d_{model} \to d_{model}$).",
            r"One base encoder layer contains $\approx 3.15$ million parameters."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=Vs87qcdm8l0"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 81,
        "title": "Masked Self-Attention | Look-Ahead Causal Masking in Decoder",
        "video_id": "m6onaKFzF94",
        "duration_str": "33m 50s",
        "core_intuition": r"""
In the Encoder, self-attention is bi-directional: every token attends to all past and future tokens (e.g., word 2 looks at word 5).
However, during text generation in the Decoder, the task is **Autoregressive**: when predicting word $t$, the future words $t+1, t+2, \dots$ have not been generated yet!
During training, we feed the entire target sentence into the decoder simultaneously to leverage GPU parallelism.
If the decoder used standard self-attention, token $t$ would simply 'cheat' and look ahead at token $t+1$ (data leakage), learning zero predictive ability!
To prevent future leakage while preserving parallel training, we use **Masked Self-Attention (Causal Masking)**.
A lower-triangular matrix masks out future positions by setting their attention logits to $-\infty$.
When passed through Softmax, $e^{-\infty} = 0$, guaranteeing that token $t$ can attend *only* to tokens $\le t$!
""",
        "key_definitions": [
            ("Masked Self-Attention (Causal Attention)", "A modified self-attention mechanism that enforces causality by masking future positions, ensuring predictions for position $t$ depend only on known outputs at positions prior to $t$."),
            ("Look-Ahead Mask (Causal Mask)", r"An upper-triangular matrix of $-\\infty$ (or zeros) used to zero out attention weights for all positions $j > i$."),
            ("Information Leakage (Cheating)", "A failure mode where an autoregressive model accesses future ground truth target tokens during training, rendering it incapable of generating text at inference.")
        ],
        "mathematical_formulations": r"""
**Mathematical Formulation of Causal Masking:**

Let raw scaled dot-product attention scores be $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$.
Define the Causal Mask $\mathbf{M} \in \mathbb{R}^{N \times N}$:

$$M_{i, j} = \begin{cases} 0 & \text{if } j \le i \quad (\text{Past and Present: Allowed}) \\ -\infty & \text{if } j > i \quad (\text{Future: Masked!}) \end{cases}$$

**Masked Attention Calculation:**
$$\mathbf{A} = \text{Softmax}(\mathbf{S} + \mathbf{M})$$

For an element in the future ($j > i$):
$$A_{i, j} = \frac{\exp(S_{i, j} + (-\infty))}{\sum_{k} \exp(S_{i, k} + M_{i, k})} = \frac{0}{\sum_{k \le i} \exp(S_{i, k})} = \mathbf{0.0}$$
Attention weight to all future tokens is strictly zero!
""",
        "architecture_and_algorithm": r"""
**Causal Mask Matrix Structure ($N=4$ tokens):**
```
Tokens:      [w1]   [w2]   [w3]   [w4]
Row 1 (w1): [  0    -inf   -inf   -inf ]  ---> Attends ONLY to w1
Row 2 (w2): [  0      0    -inf   -inf ]  ---> Attends to w1, w2
Row 3 (w3): [  0      0      0    -inf ]  ---> Attends to w1, w2, w3
Row 4 (w4): [  0      0      0      0  ]  ---> Attends to w1, w2, w3, w4

After Softmax:
Row 1:      [ 1.0    0.0    0.0    0.0 ]
Row 2:      [ 0.4    0.6    0.0    0.0 ]
Row 3:      [ 0.2    0.3    0.5    0.0 ]
Row 4:      [ 0.1    0.2    0.3    0.4 ]
```
""",
        "code_snippet": r"""```python
import tensorflow as tf

def create_causal_mask(seq_len):
    # Upper triangular matrix with ones above main diagonal
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    # Convert ones to -1e9 (-infinity)
    return mask * -1e9

# Example for seq_len = 4
mask = create_causal_mask(4)
print("Causal Mask:\n", mask.numpy())
```""",
        "exam_viva_qa": [
            (r"Why is $-\infty$ used in the mask rather than $0$ before Softmax?", 
             r"Softmax exponentiates its inputs: $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$. If you added $0$, $e^0 = 1$, which would give future tokens positive probability! Adding $-\infty$ ensures $e^{-\infty} = 0$, guaranteeing the attention weight for future tokens is absolute zero."),
            ("Why is Causal Masking unnecessary in the Encoder?", 
             "The Encoder's job is full sentence comprehension (representation learning). It has access to the entire input sentence simultaneously, and words need bidirectional context (future and past words) to disambiguate meaning. Causal masking is required *only* in autoregressive Decoders."),
            ("How does Causal Masking enable parallel training for generative models?", 
             "Without masking, an autoregressive model would have to be trained sequentially one token at a time (like an RNN). With causal masking, all $T$ target tokens can be fed into the GPU simultaneously; the mask enforces that step $t$ cannot see step $t+1$, allowing all $T$ predictions to be computed and loss evaluated in a single parallel pass!")
        ],
        "exam_takeaways": [
            r"Causal masking adds $-\infty$ to upper triangle ($j > i$).",
            r"Ensures $e^{-\infty} = 0$ in Softmax; future attention weights are zero.",
            "Enables parallel training of autoregressive models."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=m6onaKFzF94"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 82,
        "title": "Cross Attention in Transformers | Connecting Encoder to Decoder",
        "video_id": "smOnJtCevoU",
        "duration_str": "31m 15s",
        "core_intuition": r"""
How does the Decoder actually consume the representations generated by the Encoder?
Through **Cross-Attention (Decoder-Encoder Attention)**!
In the Transformer Decoder, Sublayer 1 (Masked Self-Attention) allows target tokens to communicate amongst themselves.
Sublayer 2 is the **Cross-Attention Layer**, which bridges the two worlds:
- The **Queries ($\mathbf{Q}$)** come from the preceding Decoder layer (representing: 'Here is what I have translated/generated so far, and here is what I need next').
- The **Keys ($\mathbf{K}$)** and **Values ($\mathbf{V}$)** come from the final output of the Encoder stack (representing: 'Here is the complete source sentence representation').
The Decoder Queries probe the Encoder Keys, computing attention weights that extract relevant Encoder Values to guide target token generation.
""",
        "key_definitions": [
            ("Cross-Attention (Encoder-Decoder Attention)", "A sub-layer in the Transformer decoder where queries originate from the decoder and keys and values originate from the encoder output."),
            ("Source-Target Alignment", "The dynamic mapping computed by cross-attention between target generation steps and source input representations."),
            ("Rectangular Attention Matrix", r"The $T_{dec} \times T_{enc}$ attention matrix generated in cross-attention, reflecting disparate source and target sequence lengths.")
        ],
        "mathematical_formulations": r"""
**Mathematical Formulation of Cross-Attention:**

Let $\mathbf{H}_{enc} \in \mathbb{R}^{T_{enc} \times d_{model}}$ be the final output of the 6th Encoder layer.
Let $\mathbf{H}_{dec}^{(1)} \in \mathbb{R}^{T_{dec} \times d_{model}}$ be the output of the Decoder's masked self-attention sublayer.

**Projections:**
$$\mathbf{Q} = \mathbf{H}_{dec}^{(1)} \mathbf{W}_Q^{cross} \in \mathbb{R}^{T_{dec} \times d_k}$$
$$\mathbf{K} = \mathbf{H}_{enc} \mathbf{W}_K^{cross} \in \mathbb{R}^{T_{enc} \times d_k}$$
$$\mathbf{V} = \mathbf{H}_{enc} \mathbf{W}_V^{cross} \in \mathbb{R}^{T_{enc} \times d_v}$$

**Cross-Attention Computation:**
$$\mathbf{A}_{cross} = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} \right) \in \mathbb{R}^{T_{dec} \times T_{enc}}$$
$$\mathbf{H}_{cross} = \mathbf{A}_{cross} \mathbf{V} \in \mathbb{R}^{T_{dec} \times d_v}$$

Row $i$ of $\mathbf{A}_{cross}$ represents how much attention the $i$-th target word pays to each of the $T_{enc}$ source words.
""",
        "architecture_and_algorithm": r"""
**Cross-Attention Bridge Diagram:**
```
ENCODER OUTPUT: H_enc (T_enc x 512)
        |
        +-----------------------------> [ W_K ] ---> Keys (T_enc x 64)   \
        |                                                                 ---> Scaled Dot Product ---> Weighted Values
        +-----------------------------> [ W_V ] ---> Values (T_enc x 64) /
                                                                                     ^
DECODER:                                                                             |
Decoder State: H_dec (T_dec x 512) ---> [ W_Q ] ---> Queries (T_dec x 64) ----------/
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

# Cross-Attention Layer in Keras
cross_attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)

# During Decoder forward pass:
# query = decoder_representations: (batch, T_dec, 512)
# key = encoder_output:            (batch, T_enc, 512)
# value = encoder_output:          (batch, T_enc, 512)
cross_out, cross_weights = cross_attention(
    query=decoder_rep,
    key=encoder_output,
    value=encoder_output,
    return_attention_scores=True
)
print("Cross Attention output shape:", cross_out.shape) # (batch, T_dec, 512)
```""",
        "exam_viva_qa": [
            ("In Cross-Attention, why do Keys and Values come from the Encoder while Queries come from the Decoder?", 
             "The Decoder is actively asking questions ('What should I translate next?' = Query). The Encoder holds the reference source information ('Here is the original sentence to look up' = Key, Value). The entity seeking information generates the Query; the repository providing information supplies Keys and Values."),
            ("Is Causal Masking applied during Cross-Attention?", 
             "**No!** The Decoder is permitted to attend to the *entire* source sentence (all $T_{enc}$ positions) at every single step. In translation, you may need to look at the very last source word when emitting the first target word (e.g., German verb-final grammar). Causal masking is used *only* in Decoder self-attention."),
            ("Does the Encoder re-run for every decoding step during inference?", 
             r"**No!** The Encoder runs exactly once on the input sentence, and its output representations $\mathbf{H}_{enc}$ are cached in memory. The Decoder queries this static cached representation at each subsequent autoregressive generation step.")
        ],
        "exam_takeaways": [
            "Queries come from Decoder; Keys & Values come from Encoder.",
            "Cross-attention is NOT causally masked (inspects full source sentence).",
            "Encoder output is cached and reused across all decoding steps."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=smOnJtCevoU"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 83,
        "title": "Transformer Decoder Architecture | Complete Layer Breakdown",
        "video_id": "DI2_hrAulYo",
        "duration_str": "44m 10s",
        "core_intuition": r"""
This lecture delivers the comprehensive, end-to-end breakdown of the **Transformer Decoder**.
While the Encoder has 2 sublayers per block, the Decoder has **3 sublayers per block**:
1. **Sublayer 1:** Masked Multi-Head Self-Attention (with look-ahead causal mask).
2. **Sublayer 2:** Cross-Attention (Multi-Head Attention over Encoder output).
3. **Sublayer 3:** Position-wise Feed-Forward Network (FFN).
Each of the 3 sublayers is enveloped by a Residual Skip Connection and Layer Normalization.
At the exit of the $N=6$ stacked Decoder blocks, the final representation passes through:
- A **Linear Projection Layer** that projects $d_{model}=512$ up to the entire target vocabulary size $V$ (e.g., $V = 32,000$ or $50,000$).
- A **Softmax layer** that produces a normalized probability distribution over all vocabulary tokens, from which the next token is sampled.
""",
        "key_definitions": [
            ("Transformer Decoder", "The autoregressive generative sub-network of the Transformer that synthesizes output sequences token-by-token using causal self-attention and cross-attention."),
            ("Linear Projection Head", r"A dense linear layer that maps the final decoder state vector $\mathbf{h} \in \mathbb{R}^{d_{model}}$ to logit scores over vocabulary size $V$: $\mathbf{z} \in \mathbb{R}^V$."),
            ("Weight Tying", r"An optimization technique (Press & Wolf, 2017) where the target embedding matrix and the final linear projection weight matrix share the exact same parameters: $\mathbf{W}_{out} = \mathbf{E}_{target}^T$, saving tens of millions of parameters.")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            ("Compare the sublayers of an Encoder block versus a Decoder block.", 
             "An **Encoder block** has 2 sublayers: (1) Unmasked Bidirectional Self-Attention, and (2) FFN. A **Decoder block** has 3 sublayers: (1) Causal Masked Self-Attention, (2) Cross-Attention over encoder output, and (3) FFN. Both employ Add & Norm around every sublayer."),
            ("Why does the Decoder contain significantly more parameters than the Encoder?", 
             r"Because of the extra Cross-Attention sublayer in every block ($1.05\text{M}$ extra parameters per layer $\times 6 = 6.3\text{M}$ params), plus the massive final linear classification projection head $\mathbf{W}_{proj} \in \mathbb{R}^{d_{model} \times V}$ which maps 512 dimensions to 32,000 vocabulary logits ($16.4\text{M}$ parameters)."),
            ("What is Weight Tying and why is it beneficial?", 
             r"Weight tying forces the target input embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$ and the final output projection matrix $\mathbf{W}_{proj}^T \in \mathbb{R}^{V \times d}$ to share the exact same memory weights. This eliminates $16-30\text{M}$ redundant parameters, regularizes the model, and guarantees that input and output semantic representations are perfectly aligned.")
        ],
        "exam_takeaways": [
            "Decoder has 3 sublayers: Masked Self-Attention, Cross-Attention, FFN.",
            r"Output linear projection maps $d_{model} \to V$ (vocabulary size).",
            "Weight tying reuses embedding matrix for final projection."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=DI2_hrAulYo"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    },
    {
        "index": 84,
        "title": "Transformer Inference | How Inference is Done Step by Step",
        "video_id": "FtsMOzlwxws",
        "duration_str": "48m 15s",
        "core_intuition": r"""
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
""",
        "key_definitions": [
            ("Autoregressive Inference", "The sequential inference process where a model generates one token at a time, appending each newly generated token to its input for the next time step."),
            ("KV Caching", r"An inference optimization where Key and Value projections from past time steps are cached in GPU memory, avoiding redundant $\mathcal{O}(N^2)$ recomputation at each generation step."),
            ("Beam Search", "A heuristic search algorithm that explores a graph by expanding the top-$B$ (beam width) most promising partial sequence candidates at each step."),
            ("Top-p (Nucleus) Sampling", "Sampling from the smallest set of top tokens whose cumulative probability exceeds threshold $p$ (e.g., $p=0.9$), dynamically expanding or contracting candidate pool based on model confidence.")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            (r"Why is Transformer training fully parallel ($\mathcal{O}(1)$ steps) while inference is sequential ($\mathcal{O}(T)$ steps)?", 
             "During training, the ground truth target sentence is already known. We feed all target tokens simultaneously and use **Causal Masking** to prevent look-ahead, evaluating loss across all tokens in one parallel pass. During inference, future tokens do not exist; each token must be generated, sampled, and fed back into the input before the next token can be predicted."),
            ("What is KV Caching and why is it mandatory in modern LLM serving?", 
             r"At step $t$, the decoder re-evaluates self-attention for all preceding tokens $1, \dots, t-1$. Computing $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ for past tokens repeatedly at every step wastes massive compute ($\mathcal{O}(T^2)$). **KV Caching** stores the Key and Value tensors of past tokens in GPU memory; at step $t$, the model only computes $\mathbf{q}_t, \mathbf{k}_t, \mathbf{v}_t$ for the newest single token and appends $\mathbf{k}_t, \mathbf{v}_t$ to the cache, slashing inference latency to $\mathcal{O}(T)$."),
            ("Compare Greedy Search, Beam Search, and Nucleus (Top-p) Sampling.", 
             "**Greedy Search** selects the single highest-probability token at each step; fast but gets stuck in repetitive loops. **Beam Search** explores the top-$B$ globally most probable sequence paths; produces precise, formal translations but sounds robotic for creative writing. **Nucleus (Top-p) Sampling** samples dynamically from the top cumulative $p$ probability mass, introducing human-like linguistic variety while preventing gibberish outliers.")
        ],
        "exam_takeaways": [
            "Inference is autoregressive (sequential) even though training is parallel.",
            r"KV Caching stores past keys and values to prevent $\mathcal{O}(T^2)$ recomputation.",
            "Beam Search is best for translation; Top-p / Top-K sampling is best for open-ended generation."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=FtsMOzlwxws"),
            ("Official Course Notes", "c:/Users/Admin/Desktop/ska/deep_learning_100_exam_notes/slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf")
        ]
    }
]
