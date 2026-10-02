"""
Module 5: Recurrent Neural Networks (RNNs), LSTMs, GRUs & Language Evolution (Lectures 55 - 67)
"""

MODULE_5_LECTURES = [
    {
        "index": 55,
        "title": "Why RNNs are Needed | Sequential Data & ANN Failures",
        "video_id": "4KpRP-YUw6c",
        "duration_str": "27m 45s",
        "core_intuition": r"""
Why can't standard feedforward ANNs or CNNs handle natural language, audio, or time-series data effectively?
Three fundamental constraints break traditional architectures on sequential data:
1. **Variable Input & Output Lengths:** A movie review or sentence can have 5 words, 50 words, or 500 words. ANNs have a fixed input dimension $D_{in}$ baked into their weight matrix $\mathbf{W} \in \mathbb{R}^{D_{in} \times H}$ and cannot accept inputs of arbitrary length.
2. **Loss of Temporal Order (Context Dependency):** In language, word order completely dictates semantic meaning:
   - "Dog bites man" vs "Man bites dog".
   - "Not bad, quite good!" vs "Not good, quite bad!".
   Bag-of-words or simple flattening treats tokens as independent orderless entities, obliterating syntax and grammar.
3. **Parameter Scaling across Timesteps:** An ANN attempting to accept 1,000 words would assign separate unique weights to word 1 versus word 100, failing to share knowledge that the verb 'runs' has the identical grammatical function regardless of where it appears in a sentence.
Recurrent Neural Networks (RNNs) solve all three problems through **Parameter Sharing across Time** and internal **Hidden Memory States**.
""",
        "key_definitions": [
            ("Sequential Data", "Any data where individual observations depend on previous observations and whose ordering conveys essential semantic information (e.g., text, audio waveforms, stock prices, DNA sequences)."),
            (r"Hidden State ($\mathbf{h}_t$)", "An internal vector representation maintained by a recurrent network that acts as memory, encoding information about the sequence seen from time step $1$ to time step $t$."),
            ("Parameter Sharing Across Time", r"The architectural constraint where the identical transition weight matrices ($\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$) are reused at every time step $t$.")
        ],
        "mathematical_formulations": r"""
**Why ANNs Fail on Variable Length Sequences:**
For an ANN:
$$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$$
$\mathbf{W}$ has fixed shape $(H, D_{in})$. If a sequence has $T$ words of embedding dimension $d$:
$$\mathbf{x} \in \mathbb{R}^{T \cdot d}$$
If $T$ changes from sample to sample, matrix multiplication $\mathbf{W}\mathbf{x}$ is undefined!

**The Recurrent Alternative (Temporal Invariance):**
Instead of one massive static matrix, an RNN defines a stationary dynamic system:
$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t; \mathbf{W})$$
Because $\mathbf{W}$ is applied recursively at every step, the network can process sequences of arbitrary length $T \in [1, \infty)$!
""",
        "architecture_and_algorithm": r"""
**Structural Comparison:**
```
Standard Feedforward (ANN):
   x1 ---> [ Dense Layer ] ---> y1    (Fixed input size, zero memory)

Recurrent Neural Network (RNN):
           +-------+
           |       | (Recurrent feedback loop: W_hh)
           v       |
   x_t ---> [ Cell h_t ] ---> y_t      (Processes arbitrary length T,
                                        maintains rolling memory h_t)
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

# In Keras, recurrent layers accept variable sequence lengths using None!
# Input shape: (batch_size, timesteps, feature_dim)
rnn_layer = layers.SimpleRNN(units=64, input_shape=(None, 100))
# 'None' means the sequence can be 5 words, 50 words, or 500 words long!
```""",
        "exam_viva_qa": [
            ("State the three primary reasons why standard ANNs are unsuitable for sequential data.", 
             "1. Inability to handle variable input/output sequence lengths. 2. Failure to model temporal order and long-term sequential dependencies. 3. Inability to share feature representations across different time steps (parameter explosion)."),
            ("How does an RNN achieve variable-length sequence processing?", 
             r"By applying the exact same recurrent cell and weight matrices ($\mathbf{W}_{xh}, \mathbf{W}_{hh}$) sequentially one token at a time. The computational graph unrolls dynamically to match the length $T$ of whatever sequence is provided."),
            ("What is the difference between Spatial Invariance in CNNs and Temporal Invariance in RNNs?", 
             "CNNs share filter weights across 2D spatial dimensions $(x, y)$ to recognize visual features anywhere in an image. RNNs share transition weights across the 1D temporal dimension $(t)$ to recognize patterns anywhere in a time sequence.")
        ],
        "exam_takeaways": [
            "Input shape to RNN: `(batch_size, timesteps, features)`.",
            "Recurrence allows processing arbitrary sequence lengths $T$.",
            r"Hidden state $\mathbf{h}_t$ serves as the network's rolling memory."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=4KpRP-YUw6c")
        ]
    },
    {
        "index": 56,
        "title": "Recurrent Neural Network | Forward Propagation & Architecture",
        "video_id": "BjWqCcbusMM",
        "duration_str": "35m 20s",
        "core_intuition": r"""
This lecture establishes the complete mathematical engine of the Vanilla Recurrent Neural Network (Elman RNN).
An RNN processes a sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$ sequentially.
At each time step $t$:
1. It receives the current input vector $\mathbf{x}_t$ and the previous hidden state vector $\mathbf{h}_{t-1}$.
2. It linearly combines them using two weight matrices: input-to-hidden $\mathbf{W}_{xh}$ and hidden-to-hidden $\mathbf{W}_{hh}$.
3. It applies a non-linear activation (almost universally **Tanh**) to produce the updated hidden memory state $\mathbf{h}_t$.
4. If an output is required at step $t$, $\mathbf{h}_t$ is projected via hidden-to-output matrix $\mathbf{W}_{hy}$ into output prediction $\hat{\mathbf{y}}_t$.
Unrolling the recurrent loop across time transforms the RNN into an equivalent deep feedforward network where depth equals the number of time steps $T$!
""",
        "key_definitions": [
            ("Vanilla / Elman RNN", "The standard recurrent architecture introduced by Jeffrey Elman in 1990 where current hidden state is a function of current input and previous hidden state."),
            ("Unrolling / Unfolding in Time", "Representing a recurrent network as a sequence of identical interconnected feedforward layers, one for each time step in the input sequence."),
            (r"Hidden-to-Hidden Weight Matrix ($\mathbf{W}_{hh}$)", r"The square transition matrix that maps the previous memory state $\mathbf{h}_{t-1}$ to the current state $\mathbf{h}_t$."),
            (r"Input-to-Hidden Weight Matrix ($\mathbf{W}_{xh}$)", r"The matrix that projects incoming feature vector $\mathbf{x}_t$ into the hidden state space.")
        ],
        "mathematical_formulations": r"""
**The Fundamental Vanilla RNN Equations:**

At time step $t \in \{1, 2, \dots, T\}$, with initial state $\mathbf{h}_0 = \mathbf{0}$:

1. **Pre-activation:**
   $$\mathbf{a}_t = \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h$$

2. **Hidden State Activation (Tanh):**
   $$\mathbf{h}_t = \tanh(\mathbf{a}_t) = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

3. **Output Prediction (Softmax / Linear):**
   $$\hat{\mathbf{y}}_t = \text{Softmax}(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y)$$

**Dimensionality Analysis:**
Let input feature dimension be $d$, hidden state dimension be $h$, and output dimension be $k$:
- $\mathbf{x}_t \in \mathbb{R}^{d \times 1}$
- $\mathbf{h}_t \in \mathbb{R}^{h \times 1}$
- $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$
- $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$
- $\mathbf{b}_h \in \mathbb{R}^{h \times 1}$
- $\mathbf{W}_{hy} \in \mathbb{R}^{k \times h}$
- $\mathbf{b}_y \in \mathbb{R}^{k \times 1}$

**Total Trainable Parameters:**
$$\text{Params} = (h \times d) + (h \times h) + h + (k \times h) + k$$
Notice that parameter count is completely independent of sequence length $T$!
""",
        "architecture_and_algorithm": r"""
**Unrolling an RNN in Time ($T=3$ steps):**
```
     y_1                   y_2                   y_3
      ^                     ^                     ^
    [W_hy]                [W_hy]                [W_hy]
      |                     |                     |
h_0 ->[ Cell ]-- W_hh ---> [ Cell ]-- W_hh ---> [ Cell ] ---> h_3
      ^                     ^                     ^
    [W_xh]                [W_xh]                [W_xh]
      |                     |                     |
     x_1                   x_2                   x_3
```
""",
        "code_snippet": r"""```python
import numpy as np

# Vanilla RNN Forward Pass from scratch in NumPy
class VanillaRNN:
    def __init__(self, d_in, d_hid, d_out):
        self.Wxh = np.random.randn(d_hid, d_in) * 0.01
        self.Whh = np.random.randn(d_hid, d_hid) * 0.01
        self.Why = np.random.randn(d_out, d_hid) * 0.01
        self.bh = np.zeros((d_hid, 1))
        self.by = np.zeros((d_out, 1))

    def forward(self, inputs):
        # inputs is a list of x_t vectors
        h = np.zeros((self.Whh.shape[0], 1))
        outputs = []
        for x in inputs:
            # h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
            h = np.tanh(np.dot(self.Whh, h) + np.dot(self.Wxh, x) + self.bh)
            y = np.dot(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h
```""",
        "exam_viva_qa": [
            ("Why is Tanh used as the activation function for hidden states in RNNs rather than ReLU?", 
             r"In an unrolled RNN of length $T$, the hidden state undergoes repeated matrix multiplications by $\mathbf{W}_{hh}$ at every single step ($h_T \approx \mathbf{W}_{hh}^T x_1$). If ReLU is used, unbounded activations ($z > 0$) can cause values to compound exponentially, leading to catastrophic numerical overflow ($+\infty$). Tanh squashes values strictly into $(-1, 1)$, keeping hidden representations bounded."),
            ("How many trainable parameters are in a `SimpleRNN(units=100)` layer with input dimension `20`?", 
             r"Using the formula: $\text{Params} = (\text{units} \times \text{input\_dim}) + (\text{units} \times \text{units}) + \text{units} = (100 \times 20) + (100 \times 100) + 100 = 2,000 + 10,000 + 100 = \mathbf{12,100}$ parameters."),
            ("What does the parameter `return_sequences=True` versus `return_sequences=False` control in Keras RNNs?", 
             r"`return_sequences=False` (default) outputs only the final hidden state vector $\mathbf{h}_T$ (shape $(M, \text{units})$), used for Many-to-One tasks like sentiment analysis. `return_sequences=True` outputs the full sequence of hidden states $(\mathbf{h}_1, \dots, \mathbf{h}_T)$ (shape $(M, T, \text{units})$), required when stacking recurrent layers or for Many-to-Many sequence tagging.")
        ],
        "exam_takeaways": [
            r"Hidden state recurrence: $\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$.",
            r"Parameters: $h \cdot d + h^2 + h$ (independent of sequence length $T$).",
            "`return_sequences=True` outputs all timesteps $(M, T, h)$; `False` outputs only final step $(M, h)$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=BjWqCcbusMM")
        ]
    },
    {
        "index": 57,
        "title": "RNN Sentiment Analysis | End-to-End Keras Code Example",
        "video_id": "JgnbwKnHMZQ",
        "duration_str": "39m 10s",
        "core_intuition": r"""
This lecture builds an end-to-end sentiment classification pipeline on the IMDb dataset (50,000 movie reviews) using Keras and SimpleRNN.
The essential NLP preprocessing and modeling pipeline:
1. **Tokenization:** Mapping raw text strings into discrete integer token indices (`Tokenizer`).
2. **Vocabulary Limiting:** Restricting to the top $V$ most frequent words (e.g., $V = 10,000$) and mapping rare words to `<OOV>` (Out-of-Vocabulary).
3. **Padding / Truncating:** Using `pad_sequences` to ensure uniform sequence length $T$ across all reviews (padding shorter reviews with zeros, truncating longer reviews).
4. **Embedding Layer:** Learning dense continuous vector representations ($\mathbf{E} \in \mathbb{R}^{V \times D}$) that capture semantic similarity (replacing high-dimensional, sparse one-hot vectors).
5. **Many-to-One Classification:** Feeding the embedded word vectors through `SimpleRNN(32)`, extracting the final state $\mathbf{h}_T$, and predicting binary sentiment with `Dense(1, activation='sigmoid')`.
""",
        "key_definitions": [
            ("Embedding Layer", r"A learnable lookup table $\mathbf{E} \in \mathbb{R}^{V \times D}$ that maps discrete integer word tokens into continuous dense semantic vectors of dimension $D$ (e.g., $D=128$)."),
            ("Sequence Padding", "Adding special padding tokens (typically zeros) to the beginning (`pre`) or end (`post`) of variable-length sequences to assemble uniform rectangular mini-batch tensors."),
            ("Many-to-One RNN", r"An RNN topology that ingests an entire multi-step sequence $(x_1, \dots, x_T)$ and emits a single classification decision at the conclusion of the sequence.")
        ],
        "mathematical_formulations": r"""
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
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            ("Why is Pre-padding (`padding='pre'`) generally superior to Post-padding (`padding='post'`) in Vanilla RNNs?", 
             r"In Many-to-One RNNs, only the final state $\mathbf{h}_T$ is passed to the classifier. If you post-pad with zeros, the final time steps are meaningless zero tokens, causing the hidden state to decay and forget the real words seen earlier. Pre-padding places zeros at the beginning, so the final time steps contain real words, leaving the memory fresh."),
            ("Why is an Embedding layer preferred over One-Hot Encoding?", 
             "One-hot vectors are high-dimensional ($10,000+$), orthogonal (cosine distance between any two words is 0, failing to capture semantic similarity), and sparse. Dense embeddings compress words into low-dimensional continuous vectors ($64-300D$) where semantically related words ('king' and 'queen') cluster close together."),
            ("What limitation did you observe when training SimpleRNN on reviews with `max_len=500`?", 
             "Accuracy plateaus around 80-84%, and training suffers from severe vanishing gradients. The network completely forgets opinions expressed at the start of a 500-word review. Solving this requires LSTMs.")
        ],
        "exam_takeaways": [
            "Use pre-padding (`padding='pre'`) for Many-to-One RNNs.",
            "Embedding layer maps token IDs to dense vectors: shape `(batch, max_len, embed_dim)`.",
            "SimpleRNN struggles on sequences longer than 50-100 steps."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=JgnbwKnHMZQ"),
            ("Colab Notebook", "https://colab.research.google.com/drive/1uY7NEHi59w4FkB8TViwLjUDKxgCA8W5G?usp=sharing")
        ]
    },
    {
        "index": 58,
        "title": "Types of RNN | One-to-One, One-to-Many, Many-to-One, Many-to-Many",
        "video_id": "TkOBxzhIySg",
        "duration_str": "26m 30s",
        "core_intuition": r"""
Recurrent Neural Networks are uniquely flexible because their input and output sequence dimensions can be decoupled.
Andrej Karpathy famously classified RNN applications into four canonical structural topologies based on input/output cardinalities:
1. **One-to-One:** Standard feedforward network (Image classification).
2. **One-to-Many:** Single input maps to variable sequence output (Image Captioning).
3. **Many-to-One:** Variable sequence input maps to single output (Sentiment Analysis, Video Action Classification).
4. **Many-to-Many (Synchronized / Same Length):** Ingests sequence, emits output at every time step (Part-of-Speech Tagging, Named Entity Recognition, Frame-by-frame video classification).
5. **Many-to-Many (Delayed / Asymmetric Length):** Ingests entire input sequence before emitting output sequence (Machine Translation, Speech Recognition, Chatbots) — the foundation of Encoder-Decoder (Seq2Seq) models!
""",
        "key_definitions": [
            ("One-to-Many RNN", "An architecture that maps a single fixed-size input into a sequential stream of outputs across multiple time steps."),
            ("Many-to-Many (Synced)", "A sequence tagging topology where input length equals output length ($T_x = T_y$), with one output prediction emitted per input token."),
            ("Seq2Seq (Asymmetric Many-to-Many)", r"An architecture composed of an Encoder reading $T_x$ inputs and a Decoder generating $T_y$ outputs ($T_x \neq T_y$).")
        ],
        "mathematical_formulations": r"""
**Mathematical Mappings for RNN Topologies:**

1. **Many-to-One:**
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t) \quad \forall t \in [1, T_x]$$
   $$\hat{\mathbf{y}} = g(\mathbf{h}_{T_x})$$

2. **Many-to-Many (Synchronized, $T_x = T_y$):**
   $$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{x}_t) \quad \forall t$$
   $$\hat{\mathbf{y}}_t = g(\mathbf{h}_t) \quad \forall t \in [1, T_x]$$
   $$\mathcal{L}_{total} = \sum_{t=1}^{T_x} \mathcal{L}(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$

3. **Many-to-Many (Delayed / Seq2Seq, $T_x \neq T_y$):**
   - Encoder: $\mathbf{h}_t^e = f_e(\mathbf{h}_{t-1}^e, \mathbf{x}_t) \implies \mathbf{c} = \mathbf{h}_{T_x}^e$ (Context Vector)
   - Decoder: $\mathbf{h}_t^d = f_d(\mathbf{h}_{t-1}^d, \mathbf{c}, \hat{\mathbf{y}}_{t-1})$
   - Predictions: $\hat{\mathbf{y}}_t = g(\mathbf{h}_t^d)$ for $t \in [1, T_y]$
""",
        "architecture_and_algorithm": r"""
**Karpathy's Taxonomy Visual Diagram:**
```
1. One-to-One     2. One-to-Many       3. Many-to-One       4. Many-to-Many (Synced)   5. Many-to-Many (Seq2Seq)
     [y]              [y1] [y2] [y3]             [y]             [y1] [y2] [y3]             [y1] [y2] [y3]
      ^                ^    ^    ^                ^               ^    ^    ^                ^    ^    ^
      |                |    |    |                |               |    |    |                |    |    |
    [ANN]            [RNN]->[RNN]->[RNN]    [RNN]->[RNN]->[RNN] [RNN]->[RNN]->[RNN]     [Enc]->[Enc] -> [Dec]->[Dec]
      ^                ^                          ^    ^    ^     ^    ^    ^             ^    ^
      |                |                          |    |    |     |    |    |             |    |
     [x]              [x]                        [x1] [x2] [x3]  [x1] [x2] [x3]          [x1] [x2]
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Many-to-One (Sentiment Analysis)
many_to_one = models.Sequential([
    layers.SimpleRNN(64, return_sequences=False, input_shape=(None, 50)),
    layers.Dense(1, activation='sigmoid')
])

# 2. Many-to-Many Synchronized (POS Tagging: TimeDistributed)
many_to_many_synced = models.Sequential([
    layers.SimpleRNN(64, return_sequences=True, input_shape=(None, 50)),
    layers.TimeDistributed(layers.Dense(15, activation='softmax')) # 15 POS tags
])
```""",
        "exam_viva_qa": [
            ("What real-world applications correspond to each of the 4 RNN topologies?", 
             "1. **One-to-Many:** Image Captioning (Input: 1 image; Output: sentence of words). 2. **Many-to-One:** Sentiment Analysis, Spoken Intent Classification. 3. **Many-to-Many (Synced):** Part-of-Speech Tagging, Named Entity Recognition, Video frame segmentation. 4. **Many-to-Many (Asymmetric):** Machine Translation (English to French), Speech-to-Text transcription."),
            ("Why can Machine Translation NOT be solved using a synchronized Many-to-Many RNN?", 
             r"Two reasons: (1) Different languages have different word counts (e.g., a 5-word English sentence may translate to 8 German words, so $T_x \neq T_y$). (2) Word order varies drastically across grammars (Subject-Verb-Object in English vs Subject-Object-Verb in German/Hindi); the network must read the *entire* sentence before emitting the first translated word."),
            ("What is the role of `TimeDistributed` in Keras?", 
             "`TimeDistributed(Dense(k))` applies the identical dense layer independently to every temporal slice of the sequence tensor $(M, T, H)$, producing output shape $(M, T, k)$ without flattening.")
        ],
        "exam_takeaways": [
            "Know all 4 topologies and real-world examples of each.",
            "Machine translation requires delayed Many-to-Many (Seq2Seq), not synchronized.",
            "Use `TimeDistributed` for synchronized sequence tagging."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=TkOBxzhIySg")
        ]
    },
    {
        "index": 59,
        "title": "Backpropagation Through Time (BPTT) | Mathematical Derivation",
        "video_id": "OvCz1acvt-k",
        "duration_str": "42m 15s",
        "core_intuition": r"""
How do you train an unrolled RNN whose parameters are shared across dozens of time steps?
The answer is **Backpropagation Through Time (BPTT)**.
When an RNN is unrolled for $T$ time steps, the total loss $\mathcal{L}$ is the sum of losses at each time step: $\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t$.
To compute the gradient with respect to the recurrent matrix $\mathbf{W}_{hh}$, we must use the Multivariable Chain Rule across time.
Because $\mathbf{W}_{hh}$ was used at step $1$, step $2$, ..., step $t$, the gradient $\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$ must sum partial derivatives accumulated all the way back from step $t$ down to step $1$!
This creates a long chain of continuous matrix multiplications $\prod_{k=j+1}^t \mathbf{W}_{hh}^T$, directly exposing why vanilla RNNs suffer from vanishing gradients.
""",
        "key_definitions": [
            ("Backpropagation Through Time (BPTT)", "The application of the backpropagation algorithm to an unrolled recurrent neural network, computing gradients by summing contributions across all time steps."),
            ("Truncated BPTT (TBPTT)", "A practical approximation that caps the backward unrolling horizon to a fixed number of steps $k$ (e.g., $k=20$), bounding compute and memory."),
            ("Temporal Jacobian Product", r"The product of Jacobian matrices $\\prod_{k=j+1}^t \\frac{\\partial \\mathbf{h}_k}{\\partial \\mathbf{h}_{k-1}}$ that transmits gradient signals across temporal intervals.")
        ],
        "mathematical_formulations": r"""
**Rigorous BPTT Derivation:**

Total Sequence Loss:
$$\mathcal{L} = \sum_{t=1}^T \mathcal{L}_t(\mathbf{y}_t, \hat{\mathbf{y}}_t)$$

The gradient with respect to $\mathbf{W}_{hh}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_{hh}} = \sum_{t=1}^T \frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}}$$

For a specific time step $t$, $\mathbf{h}_t$ depends on $\mathbf{W}_{hh}$ directly *and* indirectly through $\mathbf{h}_{t-1}$:
$$\frac{\partial \mathcal{L}_t}{\partial \mathbf{W}_{hh}} = \sum_{j=1}^t \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \cdot \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j} \cdot \frac{\partial^+ \mathbf{h}_j}{\partial \mathbf{W}_{hh}}$$

Where the temporal chain $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j}$ is a product of Jacobians:
$$\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_j} = \prod_{k=j+1}^t \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=j+1}^t \mathbf{W}_{hh}^T \text{diag}\left( 1 - \mathbf{h}_k^2 \right)$$

And $\frac{\partial^+ \mathbf{h}_j}{\partial \mathbf{W}_{hh}} = \mathbf{h}_{j-1}^T$ represents the direct local derivative.
""",
        "architecture_and_algorithm": r"""
**BPTT Computational Backward Sweep:**
```
Loss L_T ------------> Loss L_{t} -------------> Loss L_1
    |                      |                        |
    v                      v                        v
dL/dh_T <--- W_hh^T -- dL/dh_t <--- W_hh^T --- dL/dh_1
    |                      |                        |
    v                      v                        v
dL/dW_hh (accumulates sum of products across all timesteps!)
```
""",
        "code_snippet": r"""```python
import numpy as np

# BPTT Gradient computation loop for Vanilla RNN
def bptt_backward(inputs, targets, states, Why, Whh, Wxh):
    dWhh = np.zeros_like(Whh)
    dWxh = np.zeros_like(Wxh)
    dWhy = np.zeros_like(Why)
    dh_next = np.zeros((Whh.shape[0], 1))
    
    # Sweep backwards through time
    for t in reversed(range(len(inputs))):
        dy = outputs[t] - targets[t] # Loss derivative
        dWhy += np.dot(dy, states[t].T)
        
        # dh accumulated from output and future hidden state
        dh = np.dot(Why.T, dy) + dh_next
        # backprop through tanh: dtanh = (1 - h^2)
        da = (1 - states[t] ** 2) * dh
        
        dWxh += np.dot(da, inputs[t].T)
        h_prev = states[t-1] if t > 0 else np.zeros_like(states[0])
        dWhh += np.dot(da, h_prev.T)
        
        # Propagate to previous hidden state
        dh_next = np.dot(Whh.T, da)
        
    return dWxh, dWhh, dWhy
```""",
        "exam_viva_qa": [
            ("Why does BPTT cause high memory usage on long sequences?", 
             r"BPTT requires storing the complete sequence of intermediate hidden state vectors $\mathbf{h}_0, \mathbf{h}_1, \dots, \mathbf{h}_T$ in GPU VRAM during the forward pass. For a sequence of length $T=1000$, memory usage is $1000\times$ higher than a single forward step. This is why **Truncated BPTT** is used to cap backprop to $20-50$ steps."),
            ("How does BPTT reveal the mathematical origin of Vanishing and Exploding Gradients?", 
             r"The derivative chain contains the term $\prod_{k=j+1}^t \mathbf{W}_{hh}^T$. If the largest eigenvalue of $\mathbf{W}_{hh}$ is $\lambda < 1$, $\lambda^{t-j} \to 0$ exponentially as temporal distance $(t - j)$ grows, causing vanishing gradients. If $\lambda > 1$, $\lambda^{t-j} \to \infty$, causing exploding gradients."),
            ("What is Truncated BPTT (TBPTT)?", 
             "An engineering compromise where forward propagation runs across the entire sequence, but backpropagation stops after a fixed number of steps $k_1$, updating weights periodically every $k_2$ steps. This prevents VRAM exhaustion and stabilizes training.")
        ],
        "exam_takeaways": [
            r"Total gradient is the sum across all time steps: $\\sum_{t=1}^T \\frac{\\partial \\mathcal{L}_t}{\\partial \\mathbf{W}}$.",
            r"Long-range gradient contains $\\prod \\mathbf{W}_{hh}^T$, leading to exponential growth or decay.",
            "Truncated BPTT bounds memory and compute."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=OvCz1acvt-k")
        ]
    },
    {
        "index": 60,
        "title": "Problems with RNN | Vanishing Gradients & Long-Term Forgetfulness",
        "video_id": "AWHSZzp96kM",
        "duration_str": "28m 10s",
        "core_intuition": r"""
Why did Vanilla RNNs fail in real-world NLP and sequential tasks before the advent of LSTMs?
This lecture analyzes the two fatal pathologies of Vanilla RNNs:
1. **Vanishing Gradient Pathology:** As derived in BPTT, transmitting error gradients across $T$ steps requires multiplying by $\prod_{k} \mathbf{W}_{hh}^T \text{diag}(1 - \mathbf{h}_k^2)$. 
   Since $\tanh' \le 1.0$ and $\mathbf{W}_{hh}$ eigenvalues are typically $< 1$, the gradient decays to absolute zero after just 10 to 15 time steps!
   Consequently, words at the beginning of a sentence have zero influence on weight updates. The network develops **severe amnesia / short-term memory**.
   - Example: In the sentence "The **clouds** in the sky are ... **white**", the distance is short (3 words); RNN succeeds.
   - Example: In "I grew up in **France**, spoke fluent Spanish, traveled ... and I speak fluent **French**", the distance is 40 words; the RNN cannot connect "France" to "French"!
2. **Exploding Gradient Pathology:** If $\mathbf{W}_{hh}$ eigenvalues exceed $1.0$, gradients explode exponentially into `NaN`, causing parameter values to jump uncontrollably.
""",
        "key_definitions": [
            ("Long-Term Dependency Problem", "The inability of vanilla RNNs to learn relationships between tokens that are separated by more than 10-15 time steps due to vanishing gradients."),
            (r"Spectral Radius ($\rho(\mathbf{W})$)", r"The maximum absolute value of the eigenvalues of a matrix: $\\rho(\\mathbf{W}) = \\max_i |\\lambda_i|$. If $\\rho(\\mathbf{W}_{hh}) < 1$, gradients vanish; if $\\rho(\\mathbf{W}_{hh}) > 1$, gradients explode."),
            ("Gradient Clipping by Norm", r"The standard remedy for exploding gradients in RNNs: if $\\|\\mathbf{g}\\| > c$, set $\\mathbf{g} \\leftarrow c \\frac{\\mathbf{g}}{\\|\\mathbf{g}\\|}$.")
        ],
        "mathematical_formulations": r"""
**Pascanu, Mikolov & Bengio's Theorem on Vanishing Gradients (2013):**

Let the temporal Jacobian be $\mathbf{J}_k = \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \mathbf{W}_{hh}^T \text{diag}(1 - \mathbf{h}_k^2)$.
The norm of the gradient signal propagating from step $t$ back to step $j$ satisfies:

$$\left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \right\| \le \left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \right\| \cdot \prod_{k=j+1}^t \|\mathbf{J}_k\| \le \left\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \right\| \cdot (\gamma)^{t-j}$$

Where $\gamma = \lambda_{\max}(\mathbf{W}_{hh}) \cdot \max|\tanh'| = \lambda_{\max}(\mathbf{W}_{hh}) \cdot 1.0$.
- **Case 1 ($\gamma < 1$):** $\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \| \to 0$ exponentially as $(t - j) \to \infty$ (**Vanishing Gradient**).
- **Case 2 ($\gamma > 1$):** $\| \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_j} \| \to \infty$ exponentially as $(t - j) \to \infty$ (**Exploding Gradient**).
""",
        "architecture_and_algorithm": r"""
**Gradient Magnitude Decay over Time Steps:**
```
Gradient Signal
 ^
 | 1.0  (Step t)
 |  *
 |   \
 |    \ 0.1 (Step t - 5)
 |     \
 |      *_________ 0.00001 (Step t - 15)  <-- Completely vanishes!
 0----------------------------------------> Backward Steps (t - j)
```
Vanilla RNNs are practically incapable of bridging temporal spans $> 15$ steps.
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Fixing exploding gradients in RNNs via Gradient Clipping
model = models.Sequential([
    layers.SimpleRNN(64, input_shape=(None, 50)),
    layers.Dense(1)
])

# clipnorm=1.0 scales gradient vector if norm exceeds 1.0
custom_opt = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
model.compile(optimizer=custom_opt, loss='mse')
```""",
        "exam_viva_qa": [
            ("Why is Gradient Clipping effective for exploding gradients, but INEFFECTIVE for vanishing gradients?", 
             r"Exploding gradients have a distinct symptom: vector norm exceeds a threshold ($\|\mathbf{g}\| > c$). Clipping truncates the vector, preserving direction and preventing numerical overflow. For vanishing gradients, the gradient has vanished to zero; you cannot multiply zero by a scalar to recover the lost directional information! Overcoming vanishing gradients requires architectural changes (LSTMs, GRUs, Residual connections)."),
            ("What is the maximum effective context memory span of a Vanilla RNN?", 
             "Empirically, vanilla RNNs can reliably retain context across only $8$ to $15$ time steps. Beyond 15 steps, the gradients decay below numerical precision ($< 10^{-7}$), making it impossible to learn dependencies across paragraphs or long audio clips."),
            (r"Why does initializing $\mathbf{W}_{hh}$ as an Identity matrix (IRNN) help mitigate vanishing gradients?", 
             r"Le et al. (2015) showed that initializing $\mathbf{W}_{hh} = \mathbf{I}$ (the identity matrix) with ReLU activations sets initial eigenvalues to $1.0$ ($\lambda = 1$). In early epochs, activations and gradients pass unchanged through time ($\mathbf{I}^T = \mathbf{I}$), mimicking constant memory flow.")
        ],
        "exam_takeaways": [
            r"Vanilla RNN context limit: $\\approx 10-15$ steps.",
            "Gradient clipping solves exploding gradients, but CANNOT solve vanishing gradients.",
            "Vanishing gradients necessitated the invention of LSTMs and GRUs."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=AWHSZzp96kM")
        ]
    },
    {
        "index": 61,
        "title": "LSTM (Long Short Term Memory) Part 1 | The What & Core Intuition",
        "video_id": "z7IPBg6MyrU",
        "duration_str": "32m 45s",
        "core_intuition": r"""
Long Short-Term Memory (LSTM) (Hochreiter & Schmidhuber, 1997) is one of the most brilliant architectural innovations in the history of artificial intelligence.
LSTMs solved the vanishing gradient problem by redesigning the recurrent neuron from the ground up.
The Core Intuition: **The Conveyor Belt / Information Superhighway (Cell State $\mathbf{C}_t$)**.
In a vanilla RNN, the hidden state is violently overwritten at every step by non-linear matrix multiplication: $\mathbf{h}_t = \tanh(\mathbf{W}\mathbf{h} + \mathbf{W}\mathbf{x})$.
In an LSTM, the long-term memory (**Cell State $\mathbf{C}_t$**) runs straight down the entire chain with only minimal linear interactions!
Information can travel down this highway completely unmodified across hundreds of time steps.
To control what enters and leaves this highway, LSTMs introduce three specialized regulatory valves called **Gates** (Forget Gate, Input Gate, Output Gate), each powered by a Sigmoid function ($\sigma \in [0, 1]$).
""",
        "key_definitions": [
            ("LSTM (Long Short-Term Memory)", "A specialized recurrent neural network architecture designed to learn long-term dependencies by regulating information flow through gated mechanisms and an additive cell state channel."),
            (r"Cell State ($\mathbf{C}_t$)", "The internal memory conveyor belt of an LSTM that transports information across time with linear, additive updates, preventing gradient decay."),
            (r"Hidden State ($\mathbf{h}_t$)", "The filtered working memory output of the LSTM cell at time $t$."),
            ("Gate", r"A mechanism composed of a Sigmoid neural net layer and an element-wise multiplication that controls the proportion of information allowed to pass ($0 = \text{block completely}, 1 = \text{pass entirely}$).")
        ],
        "mathematical_formulations": r"""
**The Fundamental Constant Error Carousel (CEC) Principle:**
Why does the Cell State eliminate vanishing gradients?
In an LSTM, the cell state update is **additive**, not multiplicative:
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \widetilde{\mathbf{C}}_t$$

When computing the derivative $\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}}$:
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = \mathbf{f}_t$$

If the forget gate $\mathbf{f}_t \approx 1$ (remember):
$$\frac{\partial \mathbf{C}_t}{\partial \mathbf{C}_{t-1}} = 1.0$$
The gradient flows across time steps by multiplying by $\mathbf{f} \approx 1.0$, completely avoiding exponential decay!
$$\prod_{k=j+1}^t \frac{\partial \mathbf{C}_k}{\partial \mathbf{C}_{k-1}} \approx 1.0^{t-j} = 1.0$$
The gradient highway remains open across hundreds of time steps!
""",
        "architecture_and_algorithm": r"""
**The Three Regulatory Gates:**
1. **Forget Gate ($\mathbf{f}_t$):** 'What old information should we discard from cell state?'
2. **Input Gate ($\mathbf{i}_t$ and $\widetilde{\mathbf{C}}_t$):** 'What new candidate information should we store in cell state?'
3. **Output Gate ($\mathbf{o}_t$):** 'What parts of the cell state should be emitted as current hidden state $\mathbf{h}_t$?'

```
           Cell State C_{t-1} ---------------------[ x ]------------------(+)--------------------> Cell State C_t
                                                     ^                     ^
                                                     |                     |
                                                [Forget Gate]         [Input Gate]
                                                     |                     |
           Hidden State h_{t-1} -\              [ Sigmoid ]     [ Sigmoid ] * [ Tanh ]
                                  +--> [Gates] --+-------------------------+---------> [ Output Gate: Sigmoid ] * Tanh(C_t) -> h_t
           Input x_t ------------/
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Instantiating an LSTM in Keras
# Input shape: (batch_size, timesteps, features)
model = models.Sequential([
    layers.LSTM(64, return_sequences=True, input_shape=(None, 100)),
    layers.LSTM(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])
model.summary()
```""",
        "exam_viva_qa": [
            ("Why is the Cell State update in LSTMs additive rather than multiplicative?", 
             r"In vanilla RNNs, hidden state updates multiply by $\mathbf{W}_{hh}$ at every step, creating exponential decay during backprop. In LSTMs, the cell state update is additive ($\mathbf{C}_t = \mathbf{f} \mathbf{C}_{t-1} + \mathbf{i} \widetilde{\mathbf{C}}$). During backprop, the gradient distributes over additions, allowing error signals to flow back through time without shrinking."),
            ("What does a gate output of 0 versus 1 mean physically?", 
             r"Gates use the Sigmoid function ($\sigma(z) \in [0, 1]$). An output of 0 means 'close the valve completely—let nothing pass'. An output of 1 means 'open the valve completely—let all information pass'. Intermediate values (e.g., 0.7) represent partial retention."),
            ("Who invented the LSTM and in what year?", 
             "Sepp Hochreiter and Jürgen Schmidhuber in their landmark 1997 paper 'Long Short-Term Memory', published in Neural Computation.")
        ],
        "exam_takeaways": [
            r"Cell state $\mathbf{C}_t$ is the constant error carousel (gradient highway).",
            r"3 gates: Forget ($\mathbf{f}_t$), Input ($\mathbf{i}_t$), Output ($\mathbf{o}_t$).",
            "Additive updates prevent the vanishing gradient problem."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=z7IPBg6MyrU")
        ]
    },
    {
        "index": 62,
        "title": "LSTM Architecture | Part 2 | Complete Step-by-Step Equations",
        "video_id": "Akv3poqqwI4",
        "duration_str": "45m 12s",
        "core_intuition": r"""
This lecture delivers the comprehensive, rigorous mathematical breakdown of every single gate and equation in the LSTM cell.
Tracing input $\mathbf{x}_t$ and previous hidden state $\mathbf{h}_{t-1}$:
1. **Forget Gate Step:** Decides what fraction of prior cell state $\mathbf{C}_{t-1}$ to erase.
2. **Input Gate Step:** Computes input gate activations $\mathbf{i}_t$ and candidate state updates $\widetilde{\mathbf{C}}_t$.
3. **Cell State Update:** Scales previous state by forget gate and adds candidate state scaled by input gate.
4. **Output Gate Step:** Computes output filter $\mathbf{o}_t$ and passes cell state through Tanh to generate $\mathbf{h}_t$.
Crucial insight: Why does an LSTM have exactly **$4\times$ the parameters** of a Vanilla RNN? Because it trains 4 separate linear layers inside each cell!
""",
        "key_definitions": [
            (r"Candidate Cell State ($\widetilde{\mathbf{C}}_t$)", "A vector of new candidate values generated by a Tanh layer that could potentially be added to the cell state."),
            ("Peephole Connections", r"An architectural variant (Gers & Schmidhuber, 2000) where gate layers are allowed to inspect the cell state $\mathbf{C}_{t-1}$ directly."),
            ("Four-Fold Parameter Scaling", r"The property where an LSTM contains $4$ sets of weight matrices ($\mathbf{W}_f, \mathbf{W}_i, \mathbf{W}_c, \mathbf{W}_o$), each of dimension $H \times (D + H)$ plus biases.")
        ],
        "mathematical_formulations": r"""
**The Canonical 6 LSTM Equations (Olah / Graves Formulation):**

Given input $\mathbf{x}_t \in \mathbb{R}^d$, previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^h$, and previous cell state $\mathbf{C}_{t-1} \in \mathbb{R}^h$:

1. **Forget Gate:**
   $$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

2. **Input Gate:**
   $$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

3. **Candidate Memory State:**
   $$\widetilde{\mathbf{C}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

4. **Updated Cell State:**
   $$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \widetilde{\mathbf{C}}_t$$

5. **Output Gate:**
   $$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

6. **Updated Hidden State:**
   $$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$$

Where $[\mathbf{h}_{t-1}, \mathbf{x}_t] \in \mathbb{R}^{h + d}$ denotes vector concatenation, and $\sigma(z) = \frac{1}{1 + e^{-z}}$.

**Trainable Parameter Formula:**
Each of the 4 gates ($\mathbf{f}, \mathbf{i}, \mathbf{c}, \mathbf{o}$) has shape:
$$\mathbf{W} \in \mathbb{R}^{h \times (d + h)}, \quad \mathbf{b} \in \mathbb{R}^h$$
$$\text{Parameters}_{LSTM} = 4 \times \left[ h \cdot (d + h) + h \right] = 4 \times \left[ (d + h + 1) \cdot h \right]$$
Exactly $4\times$ the parameter footprint of a SimpleRNN!
""",
        "architecture_and_algorithm": r"""
**Complete LSTM Parameter Matrix Accounting Table:**
For `LSTM(units=128)` with input feature dimension $d = 32$:
- Hidden size $h = 128$
- Input size $d = 32$
- Single linear layer size: $(32 + 128 + 1) \times 128 = 161 \times 128 = 20,608$
- Across all 4 internal operations: $4 \times 20,608 = \mathbf{82,432 \text{ Parameters!}}$
""",
        "code_snippet": r"""```python
import numpy as np

# Pure NumPy Implementation of LSTM Cell Step
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_step(x_t, h_prev, C_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    # Concatenate h_{t-1} and x_t
    concat = np.vstack((h_prev, x_t))
    
    # 1. Gates evaluation
    f_t = sigmoid(np.dot(Wf, concat) + bf)
    i_t = sigmoid(np.dot(Wi, concat) + bi)
    c_cand = np.tanh(np.dot(Wc, concat) + bc)
    o_t = sigmoid(np.dot(Wo, concat) + bo)
    
    # 2. Cell state update
    C_t = f_t * C_prev + i_t * c_cand
    
    # 3. Hidden state update
    h_t = o_t * np.tanh(C_t)
    
    return h_t, C_t
```""",
        "exam_viva_qa": [
            (r"Why is Tanh used to generate the candidate state $\widetilde{\mathbf{C}}_t$ while Sigmoid is used for gates?", 
             r"Gates act as binary/proportional valves: they dictate *how much* information to let through, requiring a bounded range of $[0, 1]$ (Sigmoid). The candidate memory state $\widetilde{\mathbf{C}}_t$ represents actual feature values that can be added or subtracted from the cell state, requiring both positive and negative values (range $(-1, 1)$), which Tanh provides."),
            ("What is the Forget Gate Bias Initialization Trick (Józefowicz et al., 2015)?", 
             r"Initializing the forget gate bias $\mathbf{b}_f$ to a large positive value (e.g., $+1.0$ or $+2.0$) rather than zero. Since $\sigma(1.0) \approx 0.73$ and $\sigma(2.0) \approx 0.88$, this forces the LSTM to default to remembering everything at the start of training, preventing accidental early forgetting."),
            ("Calculate the parameter count of an `LSTM(units=64)` layer receiving inputs of shape `(batch, 50, 10)`.", 
             r"Using the formula: $\text{Params} = 4 \times [h(d + h + 1)] = 4 \times [64 \times (10 + 64 + 1)] = 4 \times [64 \times 75] = 4 \times 4,800 = \mathbf{19,200}$ parameters.")
        ],
        "exam_takeaways": [
            r"Formula to memorize: $\text{Params} = 4 \times [h(d + h + 1)]$.",
            "Gates use Sigmoid (range 0 to 1); candidates and states use Tanh (range -1 to 1).",
            r"Initialize forget gate bias $\mathbf{b}_f = 1.0$ to prevent early amnesia."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=Akv3poqqwI4")
        ]
    },
    {
        "index": 63,
        "title": "LSTM Part 3 | Next Word Predictor & Text Generation Project",
        "video_id": "fiqo6uPCJVI",
        "duration_str": "44m 30s",
        "core_intuition": r"""
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
""",
        "key_definitions": [
            ("Language Modeling", r"The task of estimating the joint probability distribution of sequences of words: $P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$."),
            ("Autoregressive Generation", "A generative process where the model predicts the next token conditioned on previously generated tokens, feeding its own output back as input for the next step."),
            ("Temperature Sampling", "A hyperparameter $T$ that scales logits before Softmax to control generation creativity ($T < 1.0$ makes text conservative/repetitive; $T > 1.0$ makes text creative/chaotic).")
        ],
        "mathematical_formulations": r"""
**Language Model Factorization via Chain Rule of Probability:**
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_{<t})$$

**Temperature-Scaled Softmax Sampling Formulation:**
Given raw output logit vector $\mathbf{z}$ and temperature hyperparameter $\tau > 0$:

$$P(w_i) = \frac{\exp(z_i / \tau)}{\sum_{j=1}^V \exp(z_j / \tau)}$$

- As $\tau \to 0$: Distribution collapses into a one-hot Argmax (Greedy Decoding / Zero Temperature).
- As $\tau = 1.0$: Standard Softmax probabilities.
- As $\tau \to \infty$: Distribution flattens into a uniform random distribution (Maximum Entropy / Chaos).
""",
        "architecture_and_algorithm": r"""
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
""",
        "code_snippet": r"""```python
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
```""",
        "exam_viva_qa": [
            (r"Why does Greedy Search ($\arg\max$) often produce repetitive and degenerate text?", 
             "Greedy search picks the locally most probable token at each step. In language, common high-probability words ('the', 'of', 'and') dominate, causing the model to quickly enter infinite loops (e.g., 'the model of the model of the model'). Sampling with moderate temperature ($0.7-0.8$) introduces stochastic variety that matches natural human language distributions."),
            ("How does an $N$-gram sequence generator format training data?", 
             "For a sentence of $K$ tokens, it generates $K-1$ training pairs: `(token[0], token[1])`, `(token[0:2], token[2])`, ..., `(token[0:K-1], token[K-1])`. All input prefixes are zero-padded to `max_len - 1` and target labels are categorical integers."),
            ("What is Perplexity in language modeling?", 
             r"Perplexity is the standard evaluation metric for language models, defined as the exponentiated cross-entropy loss: $\text{PPL} = \exp(\mathcal{L}_{CE}) = \exp\left(-\frac{1}{N}\sum \ln P(w_i \mid w_{<i})\right)$. A lower perplexity indicates the model is less surprised by real test text.")
        ],
        "exam_takeaways": [
            "Autoregressive generation feeds predicted tokens back as next inputs.",
            "Temperature scales logits: low $T$ = conservative, high $T$ = creative.",
            r"Perplexity $\text{PPL} = \exp(\mathcal{L}_{CE})$ measures language model quality."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=fiqo6uPCJVI"),
            ("Colab Notebook", "https://colab.research.google.com/drive/1e55Lnl0I0gFgzrbOwEAGsqmnKKwAWpRO?usp=sharing")
        ]
    },
    {
        "index": 64,
        "title": "Gated Recurrent Unit (GRU) | Modern Simplified Recurrent Cell",
        "video_id": "QQfZAoNGQmE",
        "duration_str": "31m 20s",
        "core_intuition": r"""
While LSTMs successfully solved vanishing gradients, their complexity is noticeable: 3 gates, a separate cell state, and $4\times$ the parameters of a vanilla RNN.
In 2014, Kyunghyun Cho et al. introduced the **Gated Recurrent Unit (GRU)** as a streamlined, faster alternative.
Key architectural simplifications of the GRU:
1. **Merges Cell State and Hidden State:** Eliminates the separate $\mathbf{C}_t$ channel; the hidden state $\mathbf{h}_t$ serves as both the memory carrier and working activation.
2. **Reduces Gates from 3 to 2:** Eliminates the separate Forget and Input gates, replacing them with a single coupled **Update Gate ($\mathbf{z}_t$)**. If the update gate decides to keep $80\%$ of old memory ($z_t = 0.8$), it automatically takes only $20\%$ of new candidate memory ($(1 - z_t) = 0.2$).
3. **Reset Gate ($\mathbf{r}_t$):** Controls how much of the previous state $\mathbf{h}_{t-1}$ to forget before computing the candidate state.
GRUs contain **25% fewer parameters** than LSTMs, train significantly faster, and achieve virtually identical empirical accuracy.
""",
        "key_definitions": [
            ("Gated Recurrent Unit (GRU)", "A recurrent neural network variant that combines forget and input gates into a single update gate and merges cell state into hidden state."),
            (r"Update Gate ($\mathbf{z}_t$)", r"A gate that controls the linear interpolation between the previous hidden state and the new candidate hidden state: $\\mathbf{h}_t = (1 - \\mathbf{z}_t) \\odot \\mathbf{h}_{t-1} + \\mathbf{z}_t \\odot \\widetilde{\\mathbf{h}}_t$."),
            (r"Reset Gate ($\mathbf{r}_t$)", "A gate that determines how much of the past hidden state should contribute to the candidate hidden state."),
            ("Coupled Gating", "The design choice where the retain factor ($1 - z$) and addition factor ($z$) sum strictly to $1.0$.")
        ],
        "mathematical_formulations": r"""
**The Canonical GRU Equations (Cho et al., 2014):**

Given input $\mathbf{x}_t \in \mathbb{R}^d$ and previous hidden state $\mathbf{h}_{t-1} \in \mathbb{R}^h$:

1. **Update Gate:**
   $$\mathbf{z}_t = \sigma(\mathbf{W}_z \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_z)$$

2. **Reset Gate:**
   $$\mathbf{r}_t = \sigma(\mathbf{W}_r \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_r)$$

3. **Candidate Hidden State:**
   $$\widetilde{\mathbf{h}}_t = \tanh(\mathbf{W}_h \cdot [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \ \mathbf{x}_t] + \mathbf{b}_h)$$
   *(Notice: $\mathbf{r}_t$ directly masks the previous hidden state!)*

4. **Final Hidden State (Linear Interpolation):**
   $$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \widetilde{\mathbf{h}}_t$$

**Parameter Count Formula:**
GRUs have only 3 sets of weight matrices ($\mathbf{z}, \mathbf{r}, \mathbf{h}$):
$$\text{Params}_{GRU} = 3 \times \left[ h \cdot (d + h + 1) \right]$$
Exactly **$75\%$ the parameter count of an LSTM**!
""",
        "architecture_and_algorithm": r"""
**LSTM vs GRU Structural Comparison:**

| Dimension | LSTM | GRU |
| :--- | :--- | :--- |
| **Number of Gates** | 3 (Forget, Input, Output) | 2 (Reset, Update) |
| **Memory Channels** | 2 ($\mathbf{C}_t$ Cell State, $\mathbf{h}_t$ Hidden State) | 1 ($\mathbf{h}_t$ Hidden State) |
| **Parameters** | $4 \times [h(d + h + 1)]$ | $3 \times [h(d + h + 1)]$ |
| **Compute Speed** | Slower (more matrix ops) | $\approx 25-30\%$ faster |
| **Memory Footprint** | Higher | Lower |
| **Performance** | Better on long, complex sequences | Equal or better on small/medium datasets |
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Building a GRU network in Keras
model = models.Sequential([
    layers.Embedding(input_dim=5000, output_dim=64, input_length=100),
    layers.GRU(64, return_sequences=False), # Fast 2-gate architecture
    layers.Dense(1, activation='sigmoid')
])
model.summary()
```""",
        "exam_viva_qa": [
            ("How does the GRU couple the forget and input gates?", 
             r"In an LSTM, the forget gate $\mathbf{f}_t$ and input gate $\mathbf{i}_t$ are computed independently, meaning the model could theoretically decide to remember 100% of old memory and add 100% of new memory. The GRU couples them into a single convex combination: $\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \widetilde{\mathbf{h}}_t$. Retention and addition strictly sum to 1."),
            ("Calculate the parameter count of a `GRU(units=64)` layer with input dimension `10`.", 
             r"Using the formula: $\text{Params} = 3 \times [h(d + h + 1)] = 3 \times [64 \times (10 + 64 + 1)] = 3 \times [64 \times 75] = 3 \times 4,800 = \mathbf{14,400}$ parameters. (Compare to 19,200 for an LSTM of identical capacity)."),
            ("When should one choose GRU over LSTM?", 
             "When computational resources or GPU memory are constrained, when working with smaller datasets where LSTMs might overfit due to higher parameter counts, or when training latency is a priority. For tasks requiring long-range tracking of syntactic state (like complex source code parsing), LSTMs often retain a slight edge.")
        ],
        "exam_takeaways": [
            r"GRU has 2 gates (Reset $\mathbf{r}_t$, Update $\mathbf{z}_t$) and NO separate cell state.",
            r"Parameters: $3 \times [h(d + h + 1)]$ (25% fewer parameters than LSTM).",
            "Trains faster and achieves comparable performance on most benchmarks."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=QQfZAoNGQmE")
        ]
    },
    {
        "index": 65,
        "title": "Deep RNNs | Stacked RNNs, Stacked LSTMs & Stacked GRUs",
        "video_id": "mlDkTrlLaio",
        "duration_str": "27m 50s",
        "core_intuition": r"""
Just as deep feedforward networks learn hierarchical representations across depth, recurrent networks can be stacked vertically to form **Deep (Stacked) RNNs**.
In a single-layer RNN, depth exists only horizontally across time.
By stacking recurrent layers:
- Layer 1 processes raw input vectors $\mathbf{x}_t$ and extracts low-level sequential patterns (e.g., character combinations, phonetic transitions).
- Layer 2 treats the hidden state sequence of Layer 1 as its input, learning higher-level temporal abstractions (e.g., word syntax, grammatical phrases).
- Layer 3 learns long-term semantic structures (e.g., discourse, narrative topic).
Critical implementation rule in Keras: **Every intermediate recurrent layer MUST have `return_sequences=True`** so it outputs a 3D temporal tensor `(batch, timesteps, units)` to feed the next recurrent layer!
""",
        "key_definitions": [
            ("Deep / Stacked RNN", "A recurrent network architecture featuring multiple recurrent layers stacked on top of each other, where the hidden state sequence of layer $l-1$ serves as the input sequence to layer $l$."),
            ("Temporal Abstraction Hierarchy", "The property where lower layers in a stacked RNN operate on rapid, high-frequency temporal changes while deeper layers capture slow, abstract, long-range semantic patterns."),
            ("`return_sequences=True`", "The Keras parameter that dictates whether a recurrent layer outputs its full sequence of hidden states across all time steps or only the final hidden state vector.")
        ],
        "mathematical_formulations": r"""
**Mathematical Formulation of Stacked Recurrent Layers:**

Let $\mathbf{h}_t^{[l]}$ denote the hidden state of layer $l$ at time step $t$:

For layer $l=1$:
$$\mathbf{h}_t^{[1]} = \text{RNN}\left(\mathbf{h}_{t-1}^{[1]}, \ \mathbf{x}_t\right)$$

For intermediate layers $l \in \{2, \dots, L\}$:
$$\mathbf{h}_t^{[l]} = \text{RNN}\left(\mathbf{h}_{t-1}^{[l]}, \ \mathbf{h}_t^{[l-1]}\right)$$

Notice that layer $l$ receives:
- Recurrent input from the past of its own layer: $\mathbf{h}_{t-1}^{[l]}$
- Feedforward input from the present of the previous layer: $\mathbf{h}_t^{[l-1]}$
""",
        "architecture_and_algorithm": r"""
**Stacked 3-Layer LSTM Diagram:**
```
Output:                           y_t
                                   ^
Layer 3 (LSTM):   h_0^[3] -> [ Cell ] ------> [ Cell ] ------> h_T^[3]
                                   ^                ^
Layer 2 (LSTM):   h_0^[2] -> [ Cell ] ------> [ Cell ] ------> h_T^[2]
                                   ^                ^
Layer 1 (LSTM):   h_0^[1] -> [ Cell ] ------> [ Cell ] ------> h_T^[1]
                                   ^                ^
Inputs:                           x_1              x_T
```
Layer 1 and Layer 2 must set `return_sequences=True`. Layer 3 sets `return_sequences=False` (for Many-to-One).
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Deep Stacked LSTM Architecture in Keras
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=200),
    # Layer 1: MUST return sequences!
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    # Layer 2: MUST return sequences!
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.3),
    # Layer 3: Final recurrent layer (Many-to-One)
    layers.LSTM(32, return_sequences=False),
    layers.Dense(1, activation='sigmoid')
])
```""",
        "exam_viva_qa": [
            ("What error occurs in Keras if you stack two LSTM layers without setting `return_sequences=True` on the first?", 
             r"A fatal tensor shape mismatch error. A standard LSTM outputs a 2D tensor of shape `(batch_size, units)` representing only the final time step $\mathbf{h}_T$. The second LSTM layer requires a 3D sequential input tensor of shape `(batch_size, timesteps, features)`. Setting `return_sequences=True` ensures the first layer outputs `(batch_size, timesteps, units)`."),
            ("Why do stacked RNNs rarely exceed 3 to 4 layers in depth?", 
             "Training stacked RNNs compounds vanishing and exploding gradients in *two* directions simultaneously: horizontally across time $T$, and vertically across layers $L$. Without residual connections, stacking beyond 3-4 recurrent layers degrades training stability."),
            ("How did Google's Neural Machine Translation (GNMT) system train an 8-layer stacked LSTM?", 
             r"Wu et al. (2016) introduced **Residual Connections between recurrent layers**: the input to layer $l$ was added directly to its output ($\mathbf{h}_t^{[l-1]} + \mathbf{h}_t^{[l]}$), enabling error gradients to bypass layers vertically without vanishing.")
        ],
        "exam_takeaways": [
            "Always set `return_sequences=True` on all intermediate recurrent layers.",
            "Only the final recurrent layer sets `return_sequences=False` (for classification).",
            "Stacked RNNs create hierarchical feature representations across depth."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=mlDkTrlLaio"),
            ("Colab Notebook", "https://colab.research.google.com/drive/1c4eN4cPxajCpFG6yr1mAUi3sV1JaGLDY?usp=sharing")
        ]
    },
    {
        "index": 66,
        "title": "Bidirectional RNN | BiLSTM & BiGRU Architectures",
        "video_id": "k2NSm3MNdYg",
        "duration_str": "29m 15s",
        "core_intuition": r"""
In standard unidirectional RNNs, the hidden state $\mathbf{h}_t$ only has access to the *past* context ($x_1, \dots, x_t$). It knows nothing about the future!
However, in many real-world NLP tasks (e.g., Named Entity Recognition, translation, sentiment analysis), understanding a word depends fundamentally on the words that come *after* it!
Consider the polysemous sentence:
- "The **bank** of the river was muddy." (Financial bank or river bank? You only know after reading "river"!)
- "He had to **bear** the heavy load." vs "The grizzly **bear** attacked."
**Bidirectional RNNs (BiRNN / BiLSTM / BiGRU)** (Schuster & Paliwal, 1997) solve this by training two independent recurrent networks in parallel:
1. A **Forward RNN** that reads the sequence from left-to-right ($t = 1 \to T$).
2. A **Backward RNN** that reads the sequence from right-to-left ($t = T \to 1$).
At each time step $t$, the forward hidden state $\overrightarrow{\mathbf{h}}_t$ and backward hidden state $\overleftarrow{\mathbf{h}}_t$ are concatenated, providing complete past and future contextual awareness.
""",
        "key_definitions": [
            ("Bidirectional RNN (BiRNN)", "A neural network architecture that combines two independent recurrent layers running in opposite directions, allowing predictions to depend on both preceding and succeeding sequence context."),
            (r"Forward Hidden State ($\overrightarrow{\mathbf{h}}_t$)", "The representation encoding sequence context from the past ($1$ to $t$)."),
            (r"Backward Hidden State ($\overleftarrow{\mathbf{h}}_t$)", "The representation encoding sequence context from the future ($T$ down to $t$)."),
            ("Context Concatenation", r"Combining both states into a single unified representation: $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2h}$.")
        ],
        "mathematical_formulations": r"""
**Mathematical Formulation of Bidirectional RNN:**

Given sequence $\mathbf{X} = (\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_T)$:

1. **Forward Hidden State Pass ($t = 1, \dots, T$):**
   $$\overrightarrow{\mathbf{h}}_t = \tanh\left( \mathbf{W}_{x\vec{h}} \mathbf{x}_t + \mathbf{W}_{\vec{h}\vec{h}} \overrightarrow{\mathbf{h}}_{t-1} + \mathbf{b}_{\vec{h}} \right)$$

2. **Backward Hidden State Pass ($t = T, \dots, 1$):**
   $$\overleftarrow{\mathbf{h}}_t = \tanh\left( \mathbf{W}_{x\overleftarrow{h}} \mathbf{x}_t + \mathbf{W}_{\overleftarrow{h}\overleftarrow{h}} \overleftarrow{\mathbf{h}}_{t+1} + \mathbf{b}_{\overleftarrow{h}} \right)$$

3. **Combined Output Representation:**
   $$\mathbf{h}_t = \left[ \overrightarrow{\mathbf{h}}_t \, ; \, \overleftarrow{\mathbf{h}}_t \right] \in \mathbb{R}^{2h}$$
   $$\hat{\mathbf{y}}_t = g\left( \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y \right)$$

**Parameter Count Formula:**
Because a Bidirectional layer instantiates two completely separate, independent recurrent cells:
$$\text{Params}_{\text{BiLSTM}} = 2 \times \text{Params}_{\text{LSTM}} = 8 \times [h(d + h + 1)]$$
Exactly double the parameters of a unidirectional cell!
""",
        "architecture_and_algorithm": r"""
**Bidirectional Flow Architecture:**
```
Forward State:   --> [-> h_1] -------> [-> h_2] -------> [-> h_3] -->
                          |                 |                 |
Combined:            [ Concat ]        [ Concat ]        [ Concat ] ---> Output y_t
                          |                 |                 |
Backward State: <-- [<- h_1] <------- [<- h_2] <------- [<- h_3] <--
                          ^                 ^                 ^
Inputs:                  x_1               x_2               x_3
```
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Bidirectional LSTM in Keras
model = models.Sequential([
    layers.Embedding(input_dim=10000, output_dim=64, input_length=150),
    # Bidirectional wrapper doubles output units: 64 forward + 64 backward = 128!
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
    layers.Dense(1, activation='sigmoid')
])
```""",
        "exam_viva_qa": [
            ("When can Bidirectional RNNs NOT be used?", 
             "In **Real-time Online Streaming tasks** and **Autoregressive Generation** (e.g., real-time speech transcription, live stock trading, next-token text prediction). In these tasks, future tokens have not occurred yet, making backward propagation physically impossible. BiRNNs require the *complete* sequence to be available upfront."),
            ("Why does a `Bidirectional(LSTM(64))` output a 128-dimensional vector?", 
             r"By default, Keras concatenates the 64-dimensional forward hidden state $\overrightarrow{\mathbf{h}}$ with the 64-dimensional backward hidden state $\overleftarrow{\mathbf{h}}$, yielding $64 + 64 = 128$ dimensions (`merge_mode='concat'`)."),
            ("What are the alternative `merge_mode` options in Keras Bidirectional layers?", 
             "`'concat'` (default, doubles dimension), `'sum'` (adds states, keeps 64 dims), `'mul'` (multiplies states), and `'ave'` (averages states).")
        ],
        "exam_takeaways": [
            r"BiRNNs combine forward and backward context ($\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t]$).",
            "Doubles the parameter count of a unidirectional layer.",
            "Cannot be used for real-time online streaming or autoregressive generation."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=k2NSm3MNdYg")
        ]
    },
    {
        "index": 67,
        "title": "Epic History of Large Language Models (LLMs) | LSTMs to ChatGPT",
        "video_id": "8fX3rOjTloc",
        "duration_str": "52m 10s",
        "core_intuition": r"""
This lecture delivers the grand historical and architectural narrative tracing the evolution of Natural Language Processing from early statistical methods to modern generative LLMs.
The major evolutionary paradigms:
1. **Statistical N-Grams & Rule-Based NLP (1980s–2000s):** Counting co-occurrences; brittle, zero semantic generalization.
2. **Static Distributed Word Embeddings (2013):** Word2Vec (Mikolov) & GloVe proved that words could be embedded into dense vectors where vector arithmetic captured semantic relationships ($\text{King} - \text{Man} + \text{Woman} \approx \text{Queen}$). But words had static vectors: "bank" had the same vector in river bank and financial bank.
3. **Recurrent Era & Seq2Seq (2014–2016):** LSTMs and GRUs enabled sequence processing, but sequential processing prevented GPU parallelization and created information bottlenecks.
4. **The Attention & Transformer Watershed (2017):** "Attention Is All You Need" (Vaswani et al.) eliminated recurrence, processing all tokens in parallel.
5. **Pretrained Foundation Models (2018–2020):**
   - **BERT (2018):** Masked Language Modeling (Bi-directional Encoder).
   - **GPT Series (2018–2020):** Causal Autoregressive Language Modeling (Decoder-only scaling laws).
6. **Instruction Tuning & RLHF (2022–Present):** InstructGPT and ChatGPT aligning foundation models to human intent.
""",
        "key_definitions": [
            ("Foundation Model", "A massive deep learning model trained on broad data at scale (typically via self-supervised pretraining) that can be adapted to a wide range of downstream tasks."),
            ("Static vs Contextualized Embeddings", "Static embeddings (Word2Vec) assign a single fixed vector to each word token; contextualized embeddings (BERT, GPT) generate dynamic vectors that shift based on surrounding sentential context."),
            ("Scaling Laws (Kaplan et al., 2020)", "Empirical power laws showing that cross-entropy loss decreases predictably as a power-law function of compute budget $C$, dataset size $D$, and parameter count $N$."),
            ("Reinforcement Learning from Human Feedback (RLHF)", "A post-training alignment technique that uses a human preference reward model and PPO optimization to align LLMs with helpfulness, accuracy, and safety.")
        ],
        "mathematical_formulations": r"""
**Chinchilla Optimal Scaling Laws (Hoffmann et al., 2022):**
For compute budget $C \approx 6 N D$ FLOPs (for a standard Transformer):
Optimal parameter count $N_{opt}$ and training tokens $D_{opt}$ should scale in equal proportion:

$$N_{opt} \propto C^{0.5}, \quad D_{opt} \propto C^{0.5}$$

To be compute-optimal, a model's token count should be approximately $20\times$ its parameter count:
$$D \approx 20 \times N$$
*(Proving that GPT-3 (175B parameters trained on 300B tokens) was significantly undertrained, leading to LLaMA and Chinchilla models trained on 1-2 Trillion tokens).*
""",
        "architecture_and_algorithm": r"""
**The Great NLP Architectural Lineage:**
```
[ Rule-Based / N-Grams ]
         |
[ Word2Vec / GloVe (2013) ]        (Static Embeddings)
         |
[ Seq2Seq / LSTMs (2014) ]         (Recurrent Sequential Bottleneck)
         |
[ Bahdanau Attention (2015) ]      (Dynamic Context Vector)
         |
[ Transformer (2017) ]             (Elimination of Recurrence; Massive Parallelism)
         |
    +----+--------------------------------+
    |                                     |
[ BERT (2018) ]                     [ GPT-1/2/3 (2018-2020) ]
(Encoder: Masked LM)                (Decoder: Causal Autoregressive)
    |                                     |
[ RoBERTa / DeBERTa ]               [ InstructGPT / RLHF (2022) ]
                                          |
                                    [ ChatGPT / GPT-4 / Modern LLMs ]
```
""",
        "code_snippet": r"""```python
# Demonstrating the conceptual difference between BERT and GPT
# BERT: Masked Language Modeling (Bi-directional autoencoding)
# "The capital of [MASK] is Paris" -> Predicts [MASK] using past and future context

# GPT: Causal Language Modeling (Unidirectional autoregressive)
# "The capital of France is" -> Predicts next token "Paris" using only past context
```""",
        "exam_viva_qa": [
            ("Why did Transformers completely replace LSTMs in modern Large Language Models?", 
             "LSTMs are fundamentally sequential: step $t$ cannot be computed until step $t-1$ completes. This sequential dependency creates an insurmountable computational barrier that prevents massive GPU parallelization. Transformers eliminate recurrence entirely, processing all $T$ tokens simultaneously via Self-Attention matrix multiplication, allowing models to scale to hundreds of billions of parameters across thousands of GPUs."),
            ("What is the primary difference between Word2Vec and BERT embeddings?", 
             "Word2Vec generates *static* embeddings: the word 'apple' has the exact same numerical vector whether used in 'apple fruit' or 'Apple iPhone'. BERT generates *contextualized* embeddings: every word passes through 12-24 self-attention layers, outputting a dynamic vector that reflects its precise semantic meaning in that specific sentence."),
            ("What is the difference between Encoder-only (BERT), Decoder-only (GPT), and Encoder-Decoder (T5) architectures?", 
             "**Encoder-only (BERT):** Bi-directional attention; excellent for classification, extraction, and comprehension tasks. **Decoder-only (GPT):** Causal masked attention (tokens attend only to past tokens); ideal for open-ended text generation. **Encoder-Decoder (T5/BART):** Processes full input sequence bi-directionally, then decodes output autoregressively; ideal for translation and summarization.")
        ],
        "exam_takeaways": [
            "LSTMs failed to scale because sequential execution prevents GPU parallelization.",
            "Transformers enabled scaling laws by processing all tokens in parallel.",
            "BERT = Encoder (Comprehension); GPT = Decoder (Generation)."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=8fX3rOjTloc")
        ]
    }
]
