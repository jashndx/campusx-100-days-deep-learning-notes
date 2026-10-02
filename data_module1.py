"""
Module 1: Foundations of Deep Learning & Multi-Layer Perceptrons (Lectures 01 - 14)
"""

MODULE_1_LECTURES = [
    {
        "index": 1,
        "title": "100 Days of Deep Learning | Course Announcement",
        "video_id": "2dH_qjc9mFg",
        "duration_str": "6m 12s",
        "core_intuition": r"""
This introductory lecture sets the vision, pedagogy, and rigorous roadmap of the 100 Days of Deep Learning initiative. 
The core philosophy is grounded in 'first-principles learning'—moving from elementary linear algebra and biological neurons to multi-layer perceptrons, convolutional networks, recurrent architectures, sequence-to-sequence models, and modern Transformer self-attention.
Key highlights emphasize that deep learning is not black magic; rather, it is hierarchical representation learning powered by multivariable calculus (chain rule), linear algebra (matrix operations), and numerical optimization (gradient descent).
""",
        "key_definitions": [
            ("Representation Learning", "A set of techniques that allows a system to automatically discover the representations needed for feature detection or classification from raw data."),
            ("Deep Learning", "A subfield of machine learning based on artificial neural networks with representation learning where multiple processing layers learn representations of data with multiple levels of abstraction."),
            ("Hierarchical Feature Extraction", "The mechanism where early layers learn low-level primitives (edges, textures) and deeper layers compose them into high-level semantic concepts (objects, concepts).")
        ],
        "mathematical_formulations": r"""
Deep learning models map input tensor $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$ to target predictions $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times D_{out}}$ through composed parameterized non-linear functions:

$$\hat{\mathbf{Y}} = f_L(f_{L-1}(\dots f_1(\mathbf{X}; \mathbf{W}_1, \mathbf{b}_1)\dots; \mathbf{W}_{L-1}, \mathbf{b}_{L-1}); \mathbf{W}_L, \mathbf{b}_L)$$

Where each layer $l$ performs an affine transformation followed by an element-wise activation function $\sigma$:
$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = \sigma(\mathbf{Z}^{[l]})$$
""",
        "architecture_and_algorithm": r"""
**Overall Course Roadmap Architecture:**
1. **Foundations (Days 1–14):** Perceptrons, Artificial Neurons, Forward Propagation, Loss Functions, Basic ANN Projects.
2. **Optimization & Regularization (Days 15–26):** Backpropagation math, Computational Graphs, Vanishing/Exploding Gradients, Regularization (L1/L2, Dropout), Early Stopping.
3. **Advanced Training (Days 27–39):** Activations (ReLU family), Weight Initializations (He, Xavier), Batch Normalization, Modern Optimizers (Momentum, RMSprop, Adam), Keras Tuner.
4. **Computer Vision & CNNs (Days 40–54):** Convolutions, Kernels, Padding, Pooling, LeNet-5, Transfer Learning, Functional API.
5. **Sequential Models & NLP (Days 55–67):** RNNs, Backpropagation Through Time (BPTT), LSTMs, GRUs, BiDirectional Models, LLM History.
6. **Attention & Transformers (Days 68–84):** Seq2Seq, Additive/Multiplicative Attention, Self-Attention, Multi-Head Attention, Positional Encoding, Full Transformer Encoder-Decoder.
""",
        "code_snippet": r"""```python
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
# Verify GPU availability for 100 Days of Deep Learning
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Available GPUs: {len(physical_devices)}")
```""",
        "exam_viva_qa": [
            ("What distinguishes Deep Learning from traditional Machine Learning?", 
             "Traditional ML relies heavily on manual feature engineering and domain expertise before applying algorithms (e.g., SVM, Random Forest). In contrast, Deep Learning performs end-to-end representation learning, extracting hierarchical feature representations directly from raw data via stacked layers of non-linear transformations."),
            ("Why did Deep Learning gain prominence recently despite neural network concepts existing since the 1950s?", 
             "Three pivotal factors converged: (1) Availability of massive labeled datasets (Big Data / ImageNet), (2) Hardware acceleration (high-throughput parallel compute on GPUs/TPUs), and (3) Algorithmic breakthroughs (ReLU activations preventing vanishing gradients, dropout regularization, Adam optimizer, residual connections)."),
            ("What is the Universal Approximation Theorem?", 
             r"It proves that a standard feedforward neural network with a single hidden layer containing a finite number of neurons with non-linear activation functions can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision, given sufficient neurons.")
        ],
        "exam_takeaways": [
            "Deep learning combines multivariable calculus, linear algebra, and gradient-based optimization.",
            "Understand the trade-off: DL requires significantly more data and compute than traditional ML, but scales far better as data volume increases."
        ],
        "resources_and_references": [
            ("Course Announcement Overview", "https://www.youtube.com/watch?v=2dH_qjc9mFg")
        ]
    },
    {
        "index": 2,
        "title": "What is Deep Learning? Deep Learning Vs Machine Learning",
        "video_id": "fHF22Wxuyw4",
        "duration_str": "21m 45s",
        "core_intuition": r"""
Deep Learning is a specialized sub-branch of Machine Learning, which itself is a subset of Artificial Intelligence.
In traditional Machine Learning, the bottleneck is 'feature extraction'—a human domain expert must decide how to represent raw pixels or text as engineered tabular numerical vectors (e.g., SIFT, HOG for images; TF-IDF for text). 
If the handcrafted features are suboptimal, the ML classifier (e.g., SVM, Decision Tree) plateaus regardless of data volume.
Deep Learning eliminates the manual feature engineering stage by fusing feature extraction and classification into a unified, end-to-end differentiable computational graph.
""",
        "key_definitions": [
            ("Handcrafted Features", "Manually designed algorithms (e.g., Edge detectors, SIFT, GLCM) used by domain experts to extract attributes from raw data before training traditional ML models."),
            ("End-to-End Learning", "A paradigm where a single neural network takes raw input (e.g., raw pixels, waveform) and directly predicts the target output without intermediate decoupled processing steps."),
            ("Feature Hierarchy", "The structural progression where low-level layers capture fine-grained spatial details, intermediate layers detect motifs and parts, and high-level layers capture holistic semantic categories.")
        ],
        "mathematical_formulations": r"""
**Performance vs Data Volume Scaling:**
For traditional ML algorithms, model performance $P$ plateaus with respect to dataset size $N$:
$$\lim_{N \to \infty} P_{ML}(N) = C_{ML}$$

For Deep Learning architectures with model capacity (parameter count) $\Theta$:
$$P_{DL}(N, \Theta) \propto \Theta^\alpha N^\beta \quad (\alpha, \beta > 0)$$
Performance continues to scale power-law fashion given sufficient network capacity and data volume.
""",
        "architecture_and_algorithm": r"""
**Comparison Matrix: ML vs DL**

| Dimension | Machine Learning (Traditional) | Deep Learning |
| :--- | :--- | :--- |
| **Data Dependency** | Performs well on small/medium tabular datasets | Requires large amounts of data to avoid overfitting |
| **Hardware Requirements** | CPU bound; lightweight | GPU/TPU bound; matrix multiplication heavy |
| **Feature Engineering** | Manual, labor-intensive, domain-expert dependent | Automated, learned via backpropagation |
| **Interpretability** | High (Decision trees, linear regression coefficients) | Low (Black-box parameter spaces) |
| **Training Time** | Seconds to hours | Hours to days/weeks |
| **Inference Latency** | Typically microseconds to milliseconds | Milliseconds to hundreds of milliseconds |
""",
        "code_snippet": r"""```python
# Demonstrating end-to-end classification pipeline vs manual extraction
import numpy as np
from tensorflow.keras import layers, models

# End-to-end Deep Learning: Raw inputs directly mapped to predictions
def create_end_to_end_dl_model(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Automated Feature Learning
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        # Classification
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
```""",
        "exam_viva_qa": [
            ("Why does traditional Machine Learning plateau when provided with massive datasets?", 
             "Traditional ML models have limited capacity (fewer trainable parameters) and rely on handcrafted feature representations that discard subtle high-order statistical correlations present in massive raw datasets."),
            ("What is the primary drawback of using Deep Learning on tabular data?", 
             "Tabular data lacks spatial or temporal inductive biases (such as translation invariance in images or sequence locality in text). Tree-based ensembles (XGBoost, LightGBM, CatBoost) typically outperform standard ANNs on tabular data due to better handling of unnormalized features, mixed data types, and discrete decision boundaries."),
            ("Explain the concept of 'Feature Representation Learning'.", 
             "Feature Representation Learning is the automatic transformation of raw inputs into intermediate numerical spaces where the distance and geometric orientation reflect semantic relationships, rendering the final classification or regression task linearly separable.")
        ],
        "exam_takeaways": [
            "Remember Andrew Ng's famous curve: DL outpaces traditional ML only when data scale $N$ is sufficiently large.",
            "Know the trade-offs: data size, hardware compute, interpretability, and execution time."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=fHF22Wxuyw4")
        ]
    },
    {
        "index": 3,
        "title": "Types of Neural Networks | History of Deep Learning | Applications",
        "video_id": "fne_UE7hDn0",
        "duration_str": "28m 10s",
        "core_intuition": r"""
Different data modalities possess distinct structural symmetries (inductive biases). 
Tabular data requires general-purpose fully connected networks (ANN). 
Spatial data (images) exhibits translation invariance and local pixel correlation, demanding Convolutional Neural Networks (CNNs). 
Sequential and temporal data (audio, text, time-series) exhibits order and temporal context, demanding Recurrent Neural Networks (RNNs/LSTMs) and Transformers.
The history of deep learning is marked by cycles of hype and AI winters (from McCulloch-Pitts, Rosenblatt's Perceptron, Minsky-Papert's XOR critique, to Rumelhart's backprop rediscovery, LeCun's LeNet, and the 2012 AlexNet watershed).
""",
        "key_definitions": [
            ("Artificial Neural Network (ANN / MLP)", "Fully connected feedforward architecture where each neuron in layer $l$ connects to every neuron in layer $l+1$; best suited for tabular and non-spatial data."),
            ("Convolutional Neural Network (CNN)", "Specialized neural network utilizing parameter sharing and spatial local receptive fields (convolution kernels) designed for 2D/3D grid-structured data like images."),
            ("Recurrent Neural Network (RNN)", "Architecture featuring cyclic hidden state connections, allowing information to persist across sequential time steps."),
            ("Inductive Bias", "The set of prior assumptions an algorithm uses to predict outputs of unseen inputs (e.g., spatial locality in CNNs, sequential ordering in RNNs).")
        ],
        "mathematical_formulations": r"""
**Taxonomy of Structural Mappings:**
- **ANN (Fully Connected Layer):**
  $$\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b}), \quad \mathbf{W} \in \mathbb{R}^{M \times N}$$
- **CNN (Discrete 2D Cross-Correlation / Convolution):**
  $$S(i, j) = (I * K)(i, j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m, n)$$
- **RNN (Recurrent State Transition):**
  $$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$
""",
        "architecture_and_algorithm": r"""
**Historical Milestones Timeline:**
1. **1943 - McCulloch-Pitts Neuron:** First mathematical abstraction of a biological neuron (binary threshold logic).
2. **1958 - Rosenblatt's Perceptron:** First learnable single-layer weight-updating machine.
3. **1969 - Minsky & Papert Critique:** Proved single-layer perceptrons cannot solve non-linear problems (XOR), triggering the first AI Winter.
4. **1986 - Rumelhart, Hinton & Williams:** Popularized Backpropagation for training multi-layer networks.
5. **1998 - Yann LeCun (LeNet-5):** Successful application of CNNs for handwritten digit recognition (check reading).
6. **2012 - AlexNet (Krizhevsky, Sutskever, Hinton):** Crushed ImageNet competition using GPUs and ReLU, igniting the modern Deep Learning revolution.
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers

# Architectural comparison in Keras
# 1. Dense (ANN)
ann_layer = layers.Dense(units=64, activation='relu')

# 2. Convolutional (CNN)
cnn_layer = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

# 3. Recurrent (RNN)
rnn_layer = layers.SimpleRNN(units=64, activation='tanh')
```""",
        "exam_viva_qa": [
            ("What caused the first AI Winter in 1969?", 
             "Marvin Minsky and Seymour Papert published the book 'Perceptrons', proving mathematically that single-layer perceptrons could not compute the simple exclusive-OR (XOR) function, and pessimistically conjectured that extending them to multi-layers would be computationally intractable to train."),
            ("Why is an ANN unsuitable for high-resolution images?", 
             "Two primary reasons: (1) Parameter Explosion: A 1000x1000 RGB image has 3 million inputs; connecting to a hidden layer of 1000 neurons requires 3 billion weights, leading to immediate out-of-memory errors and extreme overfitting. (2) Loss of Spatial Structure: Flattening a 2D image into a 1D vector completely destroys 2D spatial locality and translation invariance."),
            ("What is the key advantage of CNN parameter sharing?", 
             "In CNNs, the same kernel filter is convolved across the entire spatial extent of the input image. This guarantees translation equivariance and drastically reduces the number of trainable parameters compared to fully connected layers.")
        ],
        "exam_takeaways": [
            "Be able to draw and explain the historical timeline from McCulloch-Pitts (1943) to AlexNet (2012).",
            "State precisely why ANN is used for tabular, CNN for image, and RNN/Transformer for sequential data."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=fne_UE7hDn0")
        ]
    },
    {
        "index": 4,
        "title": "What is a Perceptron? Perceptron Vs Neuron | Geometric Intuition",
        "video_id": "X7iIKPoZ0Sw",
        "duration_str": "24m 50s",
        "core_intuition": r"""
The Perceptron is the foundational building block of Artificial Neural Networks. 
Biologically inspired by the neuron: dendrites receive electrical signals ($x_i$), the cell body (soma) accumulates and sums weighted inputs ($\sum w_i x_i + b$), and the axon fires an action potential if the sum breaches a threshold.
Geometrically, a perceptron defines a linear hyperplane in $d$-dimensional space that separates two classes:
- In 2D space: a straight line ($w_1 x_1 + w_2 x_2 + b = 0$).
- In 3D space: a 2D plane ($w_1 x_1 + w_2 x_2 + w_3 x_3 + b = 0$).
- In $n$-D space: an $(n-1)$-dimensional hyperplane ($\mathbf{w}^T \mathbf{x} + b = 0$).
Points on one side yield positive dot products (Class 1), while points on the other yield negative dot products (Class 0).
""",
        "key_definitions": [
            ("Perceptron", "A binary linear classification algorithm invented by Frank Rosenblatt that maps real-valued input vectors to binary output decisions via a step activation function."),
            ("Hyperplane", "An affine subspace of dimension $k-1$ in a $k$-dimensional vector space that divides the space into two disconnected half-spaces."),
            ("Linear Separability", "A geometric property where two sets of points can be completely segregated by at least one flat hyperplane without misclassifying any instance.")
        ],
        "mathematical_formulations": r"""
**Perceptron Forward Pass:**
Given input vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ and weight vector $\mathbf{w} = [w_1, w_2, \dots, w_d]^T$ with bias $b$:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{i=1}^d w_i x_i + b$$

**Step Activation Function:**
$$\hat{y} = f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}$$

**Geometric Distance to Decision Boundary:**
The signed perpendicular Euclidean distance from any point $\mathbf{x}_i$ to the separating hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ is:
$$d(\mathbf{x}_i) = \frac{\mathbf{w}^T \mathbf{x}_i + b}{\|\mathbf{w}\|_2} = \frac{\mathbf{w}^T \mathbf{x}_i + b}{\sqrt{\sum_{j=1}^d w_j^2}}$$
""",
        "architecture_and_algorithm": r"""
```
   x1 ----(w1)----\
   x2 ----(w2)-----> [ Sum: z = w^T x + b ] ---> [ Step Function f(z) ] ---> Output y_hat in {0, 1}
   xd ----(wd)----/           ^
                             |
                           Bias b
```
**Decision Boundary Properties:**
- If $\mathbf{w}^T \mathbf{x} + b > 0 \implies \hat{y} = 1$ (Positive Half-Space).
- If $\mathbf{w}^T \mathbf{x} + b < 0 \implies \hat{y} = 0$ (Negative Half-Space).
- If $\mathbf{w}^T \mathbf{x} + b = 0 \implies$ Points lie exactly on the decision boundary.
""",
        "code_snippet": r"""```python
import numpy as np

class Perceptron:
    def __init__(self, input_dim):
        self.weights = np.zeros(input_dim)
        self.bias = 0.0

    def predict(self, x):
        # Linear dot product
        z = np.dot(x, self.weights) + self.bias
        # Step activation function
        return 1 if z >= 0 else 0
```""",
        "exam_viva_qa": [
            ("What is the geometric role of the bias term $b$ in a perceptron?", 
             r"The bias term shifts the decision boundary hyperplane away from the origin. Without a bias ($b=0$), the hyperplane is constrained to pass through the coordinate origin $(0, 0, \dots, 0)$, severely limiting its ability to separate linearly separable datasets that do not center at the origin."),
            (r"What does the weight vector $\mathbf{w}$ represent geometrically?", 
             r"The weight vector $\mathbf{w}$ is the normal (perpendicular) vector to the separating hyperplane. It points in the direction of the positive half-space where $\hat{y} = 1$."),
            ("Why is the step function problematic for gradient descent?", 
             r"The standard Heaviside step function is non-differentiable at $z=0$ and has a derivative of zero everywhere else ($\frac{df}{dz} = 0 \ \forall z \neq 0$). Under gradient descent, gradients would vanish immediately, making backpropagation impossible.")
        ],
        "exam_takeaways": [
            r"Hyperplane equation: $\mathbf{w}^T \mathbf{x} + b = 0$.",
            r"Weight vector $\mathbf{w}$ is orthogonal to the decision boundary line/plane.",
            "Perceptron can only solve strictly linearly separable problems."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=X7iIKPoZ0Sw"),
            ("CampusX Day 3 Repo", "https://github.com/campusx-official/100-days-of-deep-learning/tree/main/day3")
        ]
    },
    {
        "index": 5,
        "title": "Perceptron Trick | How to train a Perceptron | Step by Step",
        "video_id": "Lu2bruOHN6g",
        "duration_str": "26m 40s",
        "core_intuition": r"""
How does a perceptron adjust its weights when it misclassifies a point?
This is demonstrated by the 'Perceptron Trick'. 
Consider a line $Ax + By + C = 0$. 
If a positive point $(p, q)$ with $y=1$ lies on the negative side (where $Ap + Bq + C < 0$), the line needs to shift towards $(p, q)$. 
To pull the line towards the point, we add the point's coordinates scaled by a learning rate $\eta$:
$A_{new} = A + \eta p, \ B_{new} = B + \eta q, \ C_{new} = C + \eta$.
If a negative point $(p, q)$ with $y=0$ lies on the positive side (where $Ap + Bq + C > 0$), we push the line away by subtracting:
$A_{new} = A - \eta p, \ B_{new} = B - \eta q, \ C_{new} = C - \eta$.
This simple algebraic operation rotates and shifts the hyperplane until all training points are correctly classified.
""",
        "key_definitions": [
            ("Perceptron Learning Rule", "An iterative weight update algorithm where weights are modified only upon encountering misclassified training samples."),
            (r"Learning Rate ($\eta$)", r"A positive scalar hyperparameter ($0 < \eta \le 1$) that controls the step magnitude of the hyperplane adjustment during each update."),
            ("Perceptron Convergence Theorem", "Block and Novikoff's mathematical proof establishing that if the training data is linearly separable, the Perceptron Learning Algorithm is guaranteed to converge in a finite number of steps.")
        ],
        "mathematical_formulations": r"""
**General Vectorized Perceptron Learning Rule:**
For a misclassified sample $(\mathbf{x}_i, y_i)$ where $y_i \in \{0, 1\}$ and $\hat{y}_i \in \{0, 1\}$:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$$
$$b^{(t+1)} = b^{(t)} + \eta (y_i - \hat{y}_i)$$

**Case Analysis:**
1. **Correctly Classified ($y_i = \hat{y}_i$):**
   $$y_i - \hat{y}_i = 0 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)}, \quad b^{(t+1)} = b^{(t)} \quad \text{(No update)}$$
2. **False Negative ($y_i = 1, \hat{y}_i = 0$):**
   $$y_i - \hat{y}_i = +1 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta \mathbf{x}_i, \quad b^{(t+1)} = b^{(t)} + \eta$$
3. **False Positive ($y_i = 0, \hat{y}_i = 1$):**
   $$y_i - \hat{y}_i = -1 \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta \mathbf{x}_i, \quad b^{(t+1)} = b^{(t)} - \eta$$
""",
        "architecture_and_algorithm": r"""
**Perceptron Training Algorithm (Epoch-by-Epoch):**
1. Initialize $\mathbf{w} \leftarrow \mathbf{0}$ or small random values, $b \leftarrow 0$.
2. For each epoch $e \in \{1, 2, \dots, \text{epochs}\}$:
   a. Set `misclassified = False`
   b. For each sample $(\mathbf{x}_i, y_i)$ in dataset:
      i. Calculate linear activation: $z_i = \mathbf{w}^T \mathbf{x}_i + b$.
      ii. Compute prediction: $\hat{y}_i = 1 \text{ if } z_i \ge 0 \text{ else } 0$.
      iii. If $\hat{y}_i \neq y_i$:
           $\mathbf{w} \leftarrow \mathbf{w} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$
           $b \leftarrow b + \eta (y_i - \hat{y}_i)$
           `misclassified = True`
   c. If not `misclassified`: Stop early (converged).
""",
        "code_snippet": r"""```python
import numpy as np

def perceptron_trick(X, y, epochs=1000, lr=0.01):
    # Add bias term column of 1s to input
    X = np.insert(X, 0, 1, axis=1)
    weights = np.ones(X.shape[1])
    
    for epoch in range(epochs):
        # Pick random point or iterate
        j = np.random.randint(0, X.shape[0])
        y_hat = 1 if np.dot(X[j], weights) >= 0 else 0
        if y[j] != y_hat:
            weights = weights + lr * (y[j] - y_hat) * X[j]
            
    return weights[0], weights[1:] # bias, weights
```""",
        "exam_viva_qa": [
            ("State the Perceptron Convergence Theorem and its prerequisite condition.", 
             r"The Perceptron Convergence Theorem states that if a dataset is linearly separable by a margin $\gamma > 0$, the perceptron algorithm will converge to a separating hyperplane in at most $k \le \left(\frac{R}{\gamma}\right)^2$ updates, where $R = \max \|\mathbf{x}_i\|$ is the radius of the data sphere."),
            ("What happens if the perceptron trick is applied to non-linearly separable data?", 
             "The algorithm will never converge. It enters an infinite loop, oscillating indefinitely between different suboptimal hyperplanes as it attempts to satisfy contradictory constraints."),
            (r"Why is the learning rate $\eta$ necessary in the update rule?", 
             r"Without $\eta$ (or if $\eta=1$), adding an entire data vector $\mathbf{x}_i$ can cause massive, erratic overshooting—violently flipping the orientation of the hyperplane and misclassifying previously correct points.")
        ],
        "exam_takeaways": [
            r"Formula to memorize: $\mathbf{w}_{new} = \mathbf{w}_{old} + \eta (y_i - \hat{y}_i) \mathbf{x}_i$.",
            "Update occurs exclusively when there is a classification error.",
            "Convergence is guaranteed ONLY for linearly separable data."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=Lu2bruOHN6g"),
            ("CampusX Day 4 Repo", "https://github.com/campusx-official/100-days-of-deep-learning/tree/main/day4")
        ]
    },
    {
        "index": 6,
        "title": "Perceptron Loss Function | Hinge Loss | Sigmoid | BCE",
        "video_id": "2_gCL5RAkHc",
        "duration_str": "35m 12s",
        "core_intuition": r"""
While the Perceptron Trick works via geometric heuristics, modern machine learning requires a differentiable 'Loss Function' that can be minimized systematically via Gradient Descent.
Why can't we use classification error count as a loss function? Because the number of misclassified points is a step-discontinuous integer function whose gradient is zero almost everywhere.
Rosenblatt proposed minimizing the sum of distances of misclassified points from the boundary.
Alternatively, replacing the step function with the smooth, differentiable **Sigmoid activation function** allows the model to output calibrated probabilities $\hat{y} \in (0, 1)$, giving rise to Logistic Regression and the **Binary Cross-Entropy (Log Loss)** loss function.
""",
        "key_definitions": [
            ("Perceptron Criterion (Loss)", r"The sum of the negative projections of misclassified samples onto the weight vector: $L(\mathbf{w}) = -\sum_{i \in \mathcal{M}} (\mathbf{w}^T \mathbf{x}_i + b) y_i$."),
            ("Sigmoid (Logistic) Function", r"A smooth S-shaped mathematical activation function $\sigma(z) = \frac{1}{1 + e^{-z}}$ that squashes any real number into the open interval $(0, 1)$."),
            ("Binary Cross-Entropy (Log Loss)", "The negative log-likelihood loss function derived from Maximum Likelihood Estimation for Bernoulli-distributed binary classification targets.")
        ],
        "mathematical_formulations": r"""
**1. Rosenblatt Perceptron Loss Formulation ($y_i \in \{-1, +1\}$):**
For misclassified samples $\mathcal{M}$, $y_i (\mathbf{w}^T \mathbf{x}_i + b) < 0$:
$$L(\mathbf{w}, b) = -\sum_{i \in \mathcal{M}} y_i (\mathbf{w}^T \mathbf{x}_i + b)$$

Gradient with respect to weights:
$$\nabla_{\mathbf{w}} L = -\sum_{i \in \mathcal{M}} y_i \mathbf{x}_i \implies \mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} + \eta \sum_{i \in \mathcal{M}} y_i \mathbf{x}_i$$

**2. Sigmoid Activation & Binary Cross Entropy ($y_i \in \{0, 1\}$):**
$$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x}_i + b)}}$$
$$\mathcal{L}_{BCE}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

Derivative of Sigmoid:
$$\frac{d\sigma(z)}{dz} = \sigma(z)(1 - \sigma(z)) = \hat{y}(1 - \hat{y})$$

Gradient of BCE Loss with respect to $w_j$:
$$\frac{\partial \mathcal{L}_{BCE}}{\partial w_j} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_{ij}$$
""",
        "architecture_and_algorithm": r"""
**Gradient Descent Optimization with BCE Loss:**
1. Initialize parameters $\mathbf{w} \sim \mathcal{N}(0, 0.01)$ and $b = 0$.
2. Compute linear combinations for batch: $\mathbf{z} = \mathbf{X}\mathbf{w} + b$.
3. Compute non-linear probability predictions: $\hat{\mathbf{y}} = \sigma(\mathbf{z})$.
4. Evaluate BCE Loss: $\mathcal{L} = -\frac{1}{N} \sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$.
5. Compute analytical gradient vector:
   $$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$
   $$\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$
6. Update parameters simultaneously:
   $$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}} \mathcal{L}, \quad b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}$$
""",
        "code_snippet": r"""```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def binary_cross_entropy(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def train_logistic_perceptron(X, y, epochs=1000, lr=0.1):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        loss = binary_cross_entropy(y, y_hat)
        
        # Gradients
        dw = (1/N) * np.dot(X.T, (y_hat - y))
        db = (1/N) * np.sum(y_hat - y)
        
        w -= lr * dw
        b -= lr * db
    return w, b
```""",
        "exam_viva_qa": [
            ("Why is 0-1 loss (misclassification count) not used with Gradient Descent?", 
             r"0-1 loss is piecewise constant. Its derivative is zero wherever it is differentiable, and undefined at the transition thresholds. Gradient descent requires informative, non-zero gradient vectors ($\nabla L \neq 0$) that indicate the directional derivative of steepest descent."),
            (r"Derive the derivative of the Sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$.", 
             r"$$\\frac{d}{dz} (1+e^{-z})^{-1} = -(1+e^{-z})^{-2} (-e^{-z}) = \\frac{e^{-z}}{(1+e^{-z})^2} = \\frac{1}{1+e^{-z}} \\cdot \\frac{e^{-z}}{1+e^{-z}} = \\sigma(z)(1 - \\sigma(z))$$"),
            ("Why is the gradient of Binary Cross-Entropy identical in form to the Mean Squared Error gradient of linear regression?", 
             r"Both belong to the Generalized Linear Model (GLM) family with canonical link functions. For the Bernoulli distribution, the canonical link is the logit function (inverse sigmoid); when paired with cross-entropy, the non-linearities in the derivative cancel out neatly to yield $(\hat{y}_i - y_i) x_{ij}$.")
        ],
        "exam_takeaways": [
            r"Know the derivation of $\\frac{d\\sigma(z)}{dz} = \\sigma(z)(1-\\sigma(z))$.",
            r"BCE Gradient: $\\nabla_w L = \\frac{1}{N} X^T (\\hat{y} - y)$.",
            r"Sigmoid maps $(-\\infty, +\\infty) \\to (0, 1)$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=2_gCL5RAkHc"),
            ("CampusX Day 5 Repo", "https://github.com/campusx-official/100-days-of-deep-learning/tree/main/day5%20-%20Perceptron%20Loss%20Function")
        ]
    },
    {
        "index": 7,
        "title": "Problem with Perceptron | Non-Linear Boundaries & XOR",
        "video_id": "Jp44b27VnOg",
        "duration_str": "19m 22s",
        "core_intuition": r"""
The fundamental mathematical limitation of the single-layer perceptron is its inability to solve non-linearly separable problems.
While logical AND, OR, and NAND can be partitioned by a single straight line, the XOR (Exclusive-OR) problem cannot.
In an XOR truth table, inputs $(0, 0)$ and $(1, 1)$ yield $0$, while $(0, 1)$ and $(1, 0)$ yield $1$. 
Plotting these four points reveals that no single 2D line can segregate the two classes.
To solve non-linear decision boundaries, one must either:
1. Manually transform the input space to higher dimensions (feature engineering / kernel trick).
2. Stack multiple perceptrons into a Multi-Layer Perceptron (MLP) with non-linear activations.
""",
        "key_definitions": [
            ("XOR Problem", "The canonical example of a non-linearly separable function that proved single-layer perceptrons cannot compute exclusive disjunction."),
            ("Convex Hull Separation", "The geometric principle stating that two sets of points are linearly separable if and only if their convex hulls do not intersect."),
            ("Multi-Layer Perceptron (MLP)", "A feedforward neural network comprising an input layer, one or more hidden layers, and an output layer, capable of learning non-linear decision surfaces.")
        ],
        "mathematical_formulations": r"""
**Proof that Single Perceptron Cannot Solve XOR:**
Assume there exist weights $w_1, w_2$ and bias $b$ such that $\hat{y} = 1 \iff w_1 x_1 + w_2 x_2 + b \ge 0$.
From the XOR truth table:
1. Input $(0, 0) \implies y=0 \implies 0 + 0 + b < 0 \implies b < 0$
2. Input $(0, 1) \implies y=1 \implies w_2 + b \ge 0$
3. Input $(1, 0) \implies y=1 \implies w_1 + b \ge 0$
4. Input $(1, 1) \implies y=0 \implies w_1 + w_2 + b < 0$

Adding inequality (2) and (3):
$$(w_1 + b) + (w_2 + b) \ge 0 \implies w_1 + w_2 + 2b \ge 0$$

Substitute inequality (4), which states $w_1 + w_2 + b < 0$:
$$(w_1 + w_2 + b) + b \ge 0 \implies \text{negative} + b \ge 0$$
Since $b < 0$ from (1), the sum of two strictly negative numbers cannot be $\ge 0$. 
This is a mathematical contradiction! Thus, no single linear perceptron can solve XOR.
""",
        "architecture_and_algorithm": r"""
**Decomposition of XOR using Multi-Layer Perceptrons:**
XOR can be expressed logically as:
$$\text{XOR}(x_1, x_2) = (x_1 \text{ OR } x_2) \text{ AND } \text{NAND}(x_1, x_2)$$

```
Layer 0 (Inputs)       Layer 1 (Hidden)               Layer 2 (Output)
   x1 -------------> [ Neuron 1: OR gate ] --------\
       \       /                                     ---> [ Neuron 3: AND gate ] ---> XOR Output
        \     /                                     /
   x2 -------------> [ Neuron 2: NAND gate ] ------/
```
By combining two linear boundaries from the hidden layer, the output layer forms a non-linear convex polygonal decision region.
""",
        "code_snippet": r"""```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Solving XOR using MLP in Keras
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

model = Sequential([
    # Hidden layer with 4 neurons and non-linear activation
    Dense(4, input_dim=2, activation='relu'),
    # Output layer
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=500, verbose=0)
print(f"XOR Predictions:\n{np.round(model.predict(X))}")
```""",
        "exam_viva_qa": [
            ("Explain why stacking multiple linear layers without activation functions fails to solve non-linear problems.", 
             r"Because the composition of linear transformations is strictly linear. If $\mathbf{y} = \mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W}'\mathbf{x} + \mathbf{b}'$. An $N$-layer network without non-linear activations collapses into a single-layer linear model."),
            ("How does a hidden layer geometrically alter the input space?", 
             "Hidden layers perform coordinate transformations—warping, bending, and projecting the input points into a new latent space where the previously entangled classes become linearly separable."),
            ("What was the historical impact of the XOR limitation on Artificial Intelligence?", 
             "Minsky and Papert's formal proof halted funding and institutional interest in neural networks for over a decade, leading to the 'First AI Winter' (1969–1980s), until backpropagation demonstrated that multi-layer perceptrons could be trained efficiently.")
        ],
        "exam_takeaways": [
            "Be able to reproduce the mathematical contradiction proof for XOR on an exam.",
            "Know that non-linear activation functions in hidden layers are what allow neural networks to bend decision boundaries."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=Jp44b27VnOg"),
            ("Colab Demonstration", "https://colab.research.google.com/drive/1x6detmf4WAUAT2pfdCts-dVrqnz4_gNB?usp=sharing")
        ]
    },
    {
        "index": 8,
        "title": "MLP Mathematical Notation | Standard Layer Indexing",
        "video_id": "H0_3SJh4Rqs",
        "duration_str": "18m 35s",
        "core_intuition": r"""
Before deriving backpropagation and multi-layer networks, establishing rigorous, standardized mathematical notation is paramount.
In deep learning literature (e.g., Goodfellow, Deep Learning Book; Andrew Ng; Michael Nielsen), ambiguous index naming is the #1 source of confusion.
This lecture codifies the standard index notation:
- Layers are indexed by superscript brackets: $[l]$ denotes layer $l$. Layer $[0]$ is input, layer $[L]$ is output.
- Weight $w_{jk}^{[l]}$ denotes the weight connecting neuron $k$ in layer $[l-1]$ to neuron $j$ in layer $[l]$.
- Bias $b_j^{[l]}$ is the bias of neuron $j$ in layer $[l]$.
- Linear combination: $z_j^{[l]}$.
- Activated output: $a_j^{[l]} = \sigma(z_j^{[l]})$.
""",
        "key_definitions": [
            (r"Weight Matrix $\mathbf{W}^{[l]}$", "A 2D tensor of shape $(n^{[l]}, n^{[l-1]})$ or $(n^{[l-1]}, n^{[l]})$ whose elements govern the affine transformation from layer $l-1$ to layer $l$."),
            (r"Activation Vector $\mathbf{A}^{[l]}$", "The vector of non-linearly transformed post-activation values emerging from layer $l$."),
            (r"Pre-activation Vector $\mathbf{Z}^{[l]}$", r"The raw linear combination vector $\mathbf{Z}^{[l]} = \mathbf{W}^{[l]} \mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$ before applying the activation function.")
        ],
        "mathematical_formulations": r"""
**Standard Deep Learning Notation Glossary:**

1. **Layer Index:**
   - $L$: Total number of layers in the network.
   - $n^{[l]}$: Number of units (neurons) in layer $l$.
   - $n^{[0]} = D_{in}$ (input dimensionality), $n^{[L]} = D_{out}$ (output classes/targets).

2. **Forward Propagation Equations (Single Sample):**
   $$z_j^{[l]} = \sum_{k=1}^{n^{[l-1]}} w_{jk}^{[l]} a_k^{[l-1]} + b_j^{[l]}$$
   $$a_j^{[l]} = g^{[l]}(z_j^{[l]})$$

3. **Matrix Vectorized Form (Batch of $M$ samples):**
   Let $\mathbf{A}^{[l-1]} \in \mathbb{R}^{M \times n^{[l-1]}}$, $\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$, $\mathbf{b}^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$:
   $$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
   $$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$
   Where $\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{M \times n^{[0]}}$.
""",
        "architecture_and_algorithm": r"""
**Dimensionality Sanity Check Table:**

| Tensor Symbol | Description | Shape (Sample-first / Keras convention) |
| :--- | :--- | :--- |
| $\mathbf{X}$ | Input Batch | $(M, n^{[0]})$ |
| $\mathbf{W}^{[l]}$ | Weights for layer $l$ | $(n^{[l-1]}, n^{[l]})$ |
| $\mathbf{b}^{[l]}$ | Biases for layer $l$ | $(1, n^{[l]})$ (broadcasted across $M$) |
| $\mathbf{Z}^{[l]}$ | Pre-activations | $(M, n^{[l]})$ |
| $\mathbf{A}^{[l]}$ | Post-activations | $(M, n^{[l]})$ |
| $\hat{\mathbf{Y}} = \mathbf{A}^{[L]}$ | Network Output | $(M, n^{[L]})$ |
""",
        "code_snippet": r"""```python
import numpy as np

# Verifying dimensional alignment for 3-layer MLP
M = 32        # Batch size
n_0 = 10      # Input features
n_1 = 64      # Hidden layer 1
n_2 = 32      # Hidden layer 2
n_3 = 1       # Output neuron

X = np.random.randn(M, n_0)
W1 = np.random.randn(n_0, n_1)
b1 = np.zeros((1, n_1))

W2 = np.random.randn(n_1, n_2)
b2 = np.zeros((1, n_2))

W3 = np.random.randn(n_2, n_3)
b3 = np.zeros((1, n_3))

# Forward pass step 1
Z1 = np.dot(X, W1) + b1
A1 = np.maximum(0, Z1) # ReLU
assert A1.shape == (M, n_1)
```""",
        "exam_viva_qa": [
            ("In the notation $w_{jk}^{[l]}$, what do $j, k,$ and $l$ denote?", 
             "$l$ denotes the target layer index. $j$ denotes the destination neuron index in layer $l$. $k$ denotes the source neuron index in the preceding layer $l-1$."),
            (r"Why is $\mathbf{b}^{[l]}$ shaped $(1, n^{[l]})$ and how does broadcasting handle batches?", 
             "Each neuron in layer $l$ has exactly one scalar bias parameter, so there are $n^{[l]}$ biases. In batch matrix multiplication, adding shape $(1, n^{[l]})$ to $(M, n^{[l]})$ automatically replicates the bias vector across all $M$ samples via NumPy/TensorFlow broadcasting."),
            ("How do you calculate the total number of trainable parameters in an MLP?", 
             r"For each layer $l$ from $1$ to $L$: $\text{Params}^{[l]} = (n^{[l-1]} \times n^{[l]}) + n^{[l]} = (n^{[l-1]} + 1) \times n^{[l]}$. Total parameters is the summation $\sum_{l=1}^L \text{Params}^{[l]}$.")
        ],
        "exam_takeaways": [
            r"Be vigilant on matrix dimension matching: $(M \times n_{l-1}) \times (n_{l-1} \times n_l) = (M \times n_l)$.",
            r"Remember total trainable parameters formula: $(n_{in} + 1) \times n_{out}$."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=H0_3SJh4Rqs")
        ]
    },
    {
        "index": 9,
        "title": "Multi Layer Perceptron | MLP Intuition & Universal Approximation",
        "video_id": "qw7wFGgNCSU",
        "duration_str": "25m 14s",
        "core_intuition": r"""
How does a collection of simple linear combinations and activations approximate any arbitrary, complex non-linear function?
This is the core intuition of the Multi-Layer Perceptron.
Geometrically, each neuron in the first hidden layer creates a single hyperplanar cut across the input space.
The second hidden layer can logically combine these cuts to construct convex polyhedral decision regions (e.g., triangles, boxes).
Additional hidden layers and neurons can combine multiple convex regions into arbitrary, disjoint, complex non-convex manifolds.
Mathematically, this intuition is codified by the Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991).
""",
        "key_definitions": [
            ("Universal Approximation Theorem", r"Theorem stating that a feedforward network with a single hidden layer containing a finite number of neurons and non-linear activations can approximate any continuous function on compact subsets of $\mathbb{R}^n$ to arbitrary precision $\epsilon > 0$."),
            ("Manifold Hypothesis", "The conjecture that high-dimensional real-world data (such as natural images or speech) concentrates near a lower-dimensional non-linear manifold embedded within the high-dimensional space."),
            ("Hidden Representation", "The intermediate feature coordinates formed by hidden layers that remap entangled raw inputs into linearly separable configurations.")
        ],
        "mathematical_formulations": r"""
**Cybenko's Theorem Statement (1989):**
Let $\sigma$ be any continuous sigmoidal activation function. 
Then finite sums of the form:

$$F(\mathbf{x}) = \sum_{i=1}^N \alpha_i \sigma(\mathbf{w}_i^T \mathbf{x} + b_i)$$

are dense in $C(I_n)$, where $I_n = [0, 1]^n$. 
In other words, given any continuous function $f \in C(I_n)$ and any tolerance $\epsilon > 0$, there exists an integer $N$ and parameters $\alpha_i, \mathbf{w}_i, b_i$ such that:

$$|F(\mathbf{x}) - f(\mathbf{x})| < \epsilon \quad \forall \mathbf{x} \in I_n$$
""",
        "architecture_and_algorithm": r"""
**Constructing Arbitrary Functions via Tower Functions:**
1. A pair of inverted sigmoids can be shifted and added to create a localized 'bump' or 'step' (tower function) in 1D.
2. In 2D, four or more hidden neurons can create a localized 2D cylinder/spike.
3. By tessellating localized bumps of varying heights ($\alpha_i$) across the input domain, an MLP acts like a multi-dimensional Riemann sum approximation of the target continuous function.
""",
        "code_snippet": r"""```python
import numpy as np
import matplotlib.pyplot as plt

# Approximating a complex 1D function using an MLP in PyTorch or Keras
import tensorflow as tf
from tensorflow.keras import layers, models

# Target non-linear function: f(x) = sin(2*pi*x) + 0.5 * cos(4*pi*x)
X_train = np.linspace(-1, 1, 500).reshape(-1, 1)
y_train = np.sin(2 * np.pi * X_train) + 0.5 * np.cos(4 * np.pi * X_train)

approximator = models.Sequential([
    layers.Dense(64, activation='tanh', input_shape=(1,)),
    layers.Dense(64, activation='tanh'),
    layers.Dense(1) # Linear output
])

approximator.compile(optimizer='adam', loss='mse')
approximator.fit(X_train, y_train, epochs=200, verbose=0)
```""",
        "exam_viva_qa": [
            ("If a single hidden layer can approximate any function, why do we use Deep (multi-layer) networks?", 
             r"The Universal Approximation Theorem guarantees *representational existence*, not *learnability* or *efficiency*. Approximating complex functions with a single hidden layer may require an exponentially large number of neurons ($\mathcal{O}(2^n)$), leading to catastrophic overfitting. Deep networks reuse hierarchical features, achieving the same expressive power with exponentially fewer parameters."),
            ("What is the difference between non-convex optimization and non-convex decision regions?", 
             r"Non-convex decision regions refer to complex geometric shapes (like concentric circles or interlocking spirals) in the input feature space. Non-convex optimization refers to the loss landscape $\mathcal{L}(\mathbf{W})$ in parameter space, which contains numerous local minima, saddle points, and ravines."),
            ("Does the Universal Approximation Theorem apply to ReLU networks?", 
             "Yes. Hornik (1991) and subsequent proofs demonstrated that the theorem holds for any non-polynomial, continuous, non-linear activation function, including ReLU.")
        ],
        "exam_takeaways": [
            "UAT proves capability, not efficiency: 1 shallow layer needs exponential width; deep layers require polynomial width.",
            "Activation function MUST be non-linear."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=qw7wFGgNCSU")
        ]
    },
    {
        "index": 10,
        "title": "Forward Propagation | How Neural Networks Predict Output",
        "video_id": "7MuiScUkboE",
        "duration_str": "31m 18s",
        "core_intuition": r"""
Forward Propagation is the deterministic computational inference pipeline of a neural network.
Data flows unidirectionally from the input layer through successive hidden layers to the output layer.
At each layer, two mathematical transformations occur:
1. An affine linear transformation: $\mathbf{Z} = \mathbf{X}\mathbf{W} + \mathbf{b}$.
2. An element-wise non-linear activation transformation: $\mathbf{A} = g(\mathbf{Z})$.
Vectorization replaces slow iterative `for` loops across individual training instances with highly parallelized BLAS (Basic Linear Algebra Subprograms) matrix multiplication executed across GPU tensor cores.
""",
        "key_definitions": [
            ("Forward Propagation", "The calculation and storage of intermediate variables (including pre-activations and activations) for a neural network in order from the first layer to the output layer."),
            ("Vectorization", "The process of rewriting scalar loop-based algorithms into matrix/tensor operations to leverage SIMD (Single Instruction Multiple Data) hardware parallelism."),
            ("Cache Storage", r"The retention of $\mathbf{Z}^{[l]}$ and $\mathbf{A}^{[l-1]}$ during the forward pass so they are readily available to compute derivatives during backpropagation.")
        ],
        "mathematical_formulations": r"""
**Comprehensive Forward Propagation Equations for Layer $l \in \{1, \dots, L\}$:**

For a mini-batch of $M$ samples:
$$\mathbf{Z}^{[l]} = \mathbf{A}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
$$\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$$

Base case (Input):
$$\mathbf{A}^{[0]} = \mathbf{X} \in \mathbb{R}^{M \times D}$$

Final Prediction (Output):
$$\hat{\mathbf{Y}} = \mathbf{A}^{[L]}$$

**Output Layer Activations by Task Type:**
1. **Binary Classification:**
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \in (0, 1)$$
2. **Multi-Class Classification ($K$ mutually exclusive classes):**
   $$\hat{y}_k = \text{Softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad \sum_{k=1}^K \hat{y}_k = 1$$
3. **Regression:**
   $$\hat{y} = z \quad (\text{Identity / Linear activation})$$
""",
        "architecture_and_algorithm": r"""
**Forward Propagation Step-by-Step Algorithm:**
1. Input tensor $\mathbf{X}$ is verified for shape $(M, D_{in})$.
2. Initialize empty dictionary `caches = []`.
3. Set $\mathbf{A}_{prev} = \mathbf{X}$.
4. For $l = 1, 2, \dots, L$:
   a. Retrieve $\mathbf{W}^{[l]}, \mathbf{b}^{[l]}, g^{[l]}$.
   b. Compute $\mathbf{Z}^{[l]} = \mathbf{A}_{prev} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$.
   c. Compute $\mathbf{A}^{[l]} = g^{[l]}(\mathbf{Z}^{[l]})$.
   d. Append tuple $(\mathbf{A}_{prev}, \mathbf{W}^{[l]}, \mathbf{b}^{[l]}, \mathbf{Z}^{[l]})$ to `caches`.
   e. Set $\mathbf{A}_{prev} = \mathbf{A}^{[l]}$.
5. Return final activation $\mathbf{A}^{[L]}$ and `caches`.
""",
        "code_snippet": r"""```python
import numpy as np

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers
    
    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']
        Z = np.dot(A_prev, W) + b
        A = np.maximum(0, Z)  # ReLU
        caches.append((A_prev, W, b, Z))
        
    # Output layer (e.g. Sigmoid for binary classification)
    W_last = parameters[f'W{L}']
    b_last = parameters[f'b{L}']
    Z_last = np.dot(A, W_last) + b_last
    AL = 1.0 / (1.0 + np.exp(-Z_last))
    caches.append((A, W_last, b_last, Z_last))
    
    return AL, caches
```""",
        "exam_viva_qa": [
            (r"Why is caching $(\mathbf{A}_{prev}, \mathbf{W}, \mathbf{Z})$ during forward propagation necessary?", 
             r"During backpropagation, computing the gradients $\\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{W}} = \\mathbf{A}_{prev}^T \\delta$ and $\\delta = \\frac{\\partial \\mathcal{L}}{\\partial \\mathbf{Z}} = \\delta_{next} \\mathbf{W}^T \\odot g'(\\mathbf{Z})$ requires the exact activation and pre-activation values computed during the forward pass. Without caching, they would need to be recomputed, doubling training time."),
            ("State the Softmax formula and explain why it is numerically unstable if implemented naively.", 
             r"Softmax is $\\sigma(z)_i = \\frac{e^{z_i}}{\\sum e^{z_j}}$. If any $z_i > 710$, floating-point overflow occurs in standard 64-bit float ($e^{710} \\approx \\infty \\implies \\text{NaN}$). The numerically stable implementation subtracts the maximum value before exponentiation: $\\frac{e^{z_i - \\max(\\mathbf{z})}}{\\sum e^{z_j - \\max(\\mathbf{z})}}$."),
            ("What is the computational complexity of forward propagation through a fully connected layer?", 
             r"Multiplying $(M \\times n_{l-1})$ by $(n_{l-1} \\times n_l)$ requires $M \\times n_{l-1} \\times n_l$ multiply-accumulate operations, giving computational complexity $\\mathcal{O}(M \\cdot n_{l-1} \\cdot n_l)$.")
        ],
        "exam_takeaways": [
            "Always memorize: Softmax for multi-class, Sigmoid for binary, Linear for regression.",
            r"Understand numeric stability trick for Softmax ($\max$ subtraction)."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=7MuiScUkboE")
        ]
    },
    {
        "index": 11,
        "title": "Customer Churn Prediction using ANN | Keras & TensorFlow",
        "video_id": "9wmImImmgcI",
        "duration_str": "42m 15s",
        "core_intuition": r"""
This lecture delivers the end-to-end industry blueprint for deploying an Artificial Neural Network on structured tabular data.
Using the classic bank customer churn dataset, the workflow covers:
1. Exploratory Data Analysis & identifying target leakage.
2. Encoding categorical attributes (One-Hot Encoding for nominal variables, Label Encoding for binary).
3. Critical step: Feature scaling using StandardScaler (neural networks will fail or train extremely slowly if features have disparate numerical scales).
4. Train/Test splitting without data snooping.
5. Model architecture design, choosing Binary Cross-Entropy loss, Adam optimizer, and monitoring validation loss and accuracy curves to detect overfitting.
""",
        "key_definitions": [
            ("Data Snooping / Leakage", "A fatal data science error where information from outside the training dataset (such as test set statistics during scaling) is inadvertently used to train the model."),
            ("One-Hot Encoding", "A transformation of categorical variables with $K$ categories into $K$ binary indicator columns."),
            ("Overfitting", "A modeling error where a neural network memorizes noise in the training set, characterized by diverging training loss (decreasing) and validation loss (increasing).")
        ],
        "mathematical_formulations": r"""
**StandardScaler Standardization Formula:**
For feature column $j$:
$$\mu_j = \frac{1}{N_{train}} \sum_{i=1}^{N_{train}} x_{ij}, \quad \sigma_j = \sqrt{\frac{1}{N_{train}} \sum_{i=1}^{N_{train}} (x_{ij} - \mu_j)^2}$$
$$x_{ij}^{scaled} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

Crucial exam rule: $\mu_j$ and $\sigma_j$ must be fitted *only* on the training split, and then used to transform both train and test splits:
$$\mathbf{X}_{test}^{scaled} = \frac{\mathbf{X}_{test} - \boldsymbol{\mu}_{train}}{\boldsymbol{\sigma}_{train}}$$
""",
        "architecture_and_algorithm": r"""
**ANN Architecture for Binary Churn Classification:**
- Input Layer: Shape $(D_{in},) = (11,)$ features after encoding.
- Hidden Layer 1: `Dense(11, activation='relu')`
- Hidden Layer 2: `Dense(11, activation='relu')`
- Output Layer: `Dense(1, activation='sigmoid')`
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metrics: `['accuracy']`
""",
        "code_snippet": r"""```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Split & Preprocess
# Assuming df loaded
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Do NOT fit on test!

# 2. Build ANN
model = Sequential([
    Dense(11, activation='relu', input_dim=X_train_scaled.shape[1]),
    Dense(11, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.2, batch_size=32)

# 3. Evaluate
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)
```""",
        "exam_viva_qa": [
            ("Why must `fit_transform` be called only on the training set and `transform` on the test set?", 
             r"Fitting on the test set causes data leakage. It pollutes the training phase with global distribution parameters ($\mu, \sigma$) of unseen evaluation data, providing overly optimistic and invalid evaluation metrics."),
            ("Why is accuracy an inadequate evaluation metric for customer churn prediction?", 
             "Customer churn datasets are typically highly class-imbalanced (e.g., 90% non-churn, 10% churn). A naive model predicting 'no churn' for all samples achieves 90% accuracy while having zero recall on the positive class. Metrics like Precision, Recall, F1-Score, and PR-AUC are required."),
            ("What does the `history` object returned by `model.fit()` contain?", 
             "It contains a dictionary (`history.history`) recording loss and evaluation metrics (e.g., `loss`, `val_loss`, `accuracy`, `val_accuracy`) evaluated at the conclusion of every single epoch, useful for diagnosing bias and variance.")
        ],
        "exam_takeaways": [
            "Never fit scalers on test data.",
            "Choose classification threshold (default 0.5) based on precision-recall trade-offs."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=9wmImImmgcI"),
            ("Kaggle Churn Notebook", "https://www.kaggle.com/campusx/notebook8ad570467f")
        ]
    },
    {
        "index": 12,
        "title": "Handwritten Digit Classification using ANN | MNIST Dataset",
        "video_id": "3xPT2Pk0Jds",
        "duration_str": "38m 40s",
        "core_intuition": r"""
This lecture tackles multi-class computer vision using an ANN on the benchmark MNIST dataset (70,000 $28 \times 28$ grayscale images of handwritten digits $0-9$).
Key pedagogical steps:
1. Understanding 2D image representations (matrices with pixel intensities $0-255$).
2. Flattening the $28 \times 28$ 2D matrix into a 784-dimensional 1D vector.
3. Normalizing pixel values by dividing by 255.0 to map inputs into $[0.0, 1.0]$.
4. Multi-class output architecture: 10 neurons with Softmax activation function.
5. Choosing between `categorical_crossentropy` (when targets are one-hot encoded) and `sparse_categorical_crossentropy` (when targets are integer class labels $0-9$).
""",
        "key_definitions": [
            ("MNIST Dataset", r"The Modified National Institute of Standards and Technology dataset consisting of 60,000 training and 10,000 testing $28 \\times 28$ grayscale handwritten digits."),
            ("Flattening", r"Reshaping a multi-dimensional array (e.g., $(28, 28)$) into a single continuous 1D vector of length $28 \\times 28 = 784$."),
            ("Sparse Categorical Cross-Entropy", "A cross-entropy loss formulation that accepts integer class labels directly without requiring one-hot encoding matrices, conserving significant memory.")
        ],
        "mathematical_formulations": r"""
**Categorical vs Sparse Categorical Cross-Entropy:**

Let true class label be $y \in \{0, 1, \dots, K-1\}$ and predicted probability vector be $\hat{\mathbf{y}} = [\hat{y}_0, \dots, \hat{y}_{K-1}]^T$:

1. **One-Hot Encoded (Categorical Cross-Entropy):**
   $$\mathbf{y}_{one\_hot} = [0, \dots, 1, \dots, 0]^T$$
   $$\mathcal{L}_{CCE} = -\sum_{k=0}^{K-1} y_k \log(\hat{y}_k)$$

2. **Integer Label (Sparse Categorical Cross-Entropy):**
   Since all terms in the summation are zero except where $k = y_{true}$:
   $$\mathcal{L}_{SCCE} = -\log(\hat{y}_{y_{true}})$$

Both yield mathematically identical gradients, but Sparse CCE saves $\mathcal{O}(N \times K)$ memory allocations.
""",
        "architecture_and_algorithm": r"""
**MNIST ANN Model Architecture:**
```
[Input: (28, 28)] 
       |
  [Flatten Layer] ---> Output: (784,)
       |
  [Dense(128, ReLU)] ---> Params: 784 * 128 + 128 = 100,480
       |
  [Dense(32, ReLU)] ---> Params: 128 * 32 + 32 = 4,128
       |
  [Dense(10, Softmax)] ---> Params: 32 * 10 + 10 = 330
```
Total Trainable Parameters: $100,480 + 4,128 + 330 = 104,938$.
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load MNIST
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Pixel Normalization
X_train, X_test = X_train / 255.0, X_test / 255.0

# 3. Model Architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.2)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
```""",
        "exam_viva_qa": [
            ("Why do we normalize pixel values from $[0, 255]$ to $[0, 1]$ before passing them to an ANN?", 
             r"Large input magnitudes (up to 255) cause the dot products $\\mathbf{z} = \\mathbf{X}\\mathbf{W} + \\mathbf{b}$ to explode, pushing activations into the saturated regions of activation functions (vanishing gradients) and creating an elongated, ill-conditioned loss surface where gradient descent oscillates wildly."),
            ("When should one use `categorical_crossentropy` versus `sparse_categorical_crossentropy` in Keras?", 
             "Use `categorical_crossentropy` when targets are one-hot encoded vectors (shape $(N, K)$). Use `sparse_categorical_crossentropy` when targets are integer scalars (shape $(N,)$). They calculate the exact same mathematical loss, but sparse CCE avoids generating large one-hot matrices."),
            ("How do you extract the predicted class label from the 10 Softmax probability outputs?", 
             "Using `np.argmax(probabilities, axis=1)`, which selects the index of the highest predicted probability.")
        ],
        "exam_takeaways": [
            "Always divide image pixels by 255.0.",
            "Flatten layer has 0 trainable parameters—it only changes tensor shape.",
            "Softmax is used at output layer for multi-class classification."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=3xPT2Pk0Jds"),
            ("Colab Notebook", "https://colab.research.google.com/drive/1SqETl3Zi1EEesdJfEv6_QimB-M-YjKGx?usp=sharing")
        ]
    },
    {
        "index": 13,
        "title": "Graduate Admission Prediction using ANN | Regression with ANN",
        "video_id": "RCmiPBiA4qg",
        "duration_str": "27m 30s",
        "core_intuition": r"""
While classification predicts discrete category probabilities, Regression predicts continuous real-valued targets (e.g., chance of graduate admission from $0.0$ to $1.0$, housing prices, stock returns).
This lecture demonstrates adapting an ANN for regression tasks:
1. Output Layer: Contains exactly 1 neuron with a **linear (identity) activation function** ($g(z) = z$) or Sigmoid if the output is strictly bounded in $[0, 1]$.
2. Loss Function: Shift from Cross-Entropy to **Mean Squared Error (MSE)** or **Mean Absolute Error (MAE)**.
3. Evaluation Metrics: $R^2$ score, RMSE, MAE.
""",
        "key_definitions": [
            ("Regression ANN", "A neural network architecture engineered to output continuous real-valued numerical variables."),
            ("Mean Squared Error (MSE)", r"The average of the squared differences between predicted values and actual ground truth targets: $\\frac{1}{N}\\sum (y - \\hat{y})^2$."),
            ("Linear Activation Function", "An identity function $f(z) = z$ that passes the weighted sum directly to the output without compression or non-linear saturation.")
        ],
        "mathematical_formulations": r"""
**Regression Loss Formulations:**

1. **Mean Squared Error (MSE / L2 Loss):**
   $$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
   Gradient with respect to prediction:
   $$\frac{\partial \mathcal{L}_{MSE}}{\partial \hat{y}_i} = -\frac{2}{N}(y_i - \hat{y}_i)$$

2. **Mean Absolute Error (MAE / L1 Loss):**
   $$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

3. **Coefficient of Determination ($R^2$ Score):**
   $$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$
""",
        "architecture_and_algorithm": r"""
**Comparison: Classification vs Regression in ANNs**

| Component | Binary Classification | Multi-Class Classification | Regression |
| :--- | :--- | :--- | :--- |
| **Output Neurons** | 1 | $K$ (number of classes) | 1 (or $M$ for multi-output) |
| **Output Activation** | `sigmoid` | `softmax` | `linear` (None) or `relu` |
| **Loss Function** | `binary_crossentropy` | `categorical_crossentropy` | `mean_squared_error` / `mae` |
| **Primary Metric** | Accuracy, ROC-AUC, F1 | Accuracy, Top-k Accuracy | MSE, RMSE, MAE, $R^2$ |
""",
        "code_snippet": r"""```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# Regression ANN Model
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')  # Output layer for continuous regression
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_scaled, y_train, epochs=100, batch_size=16, validation_split=0.2)

y_pred = model.predict(X_test_scaled)
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
```""",
        "exam_viva_qa": [
            ("Why must the output activation function be 'linear' for general regression?", 
             r"A linear activation $g(z) = z$ has an unbounded range $(-\\infty, +\\infty)$, allowing the network to predict any arbitrary real value. Using Sigmoid would bound outputs to $(0, 1)$ and ReLU would prevent negative predictions."),
            ("Compare MSE and MAE loss functions in the presence of extreme outliers.", 
             r"MSE squares the error term $(y_i - \hat{y}_i)^2$, causing outliers to dominate the gradient and pull the model heavily towards anomalies. MAE penalizes errors linearly $|y_i - \hat{y}_i|$, making it much more robust to outliers."),
            ("What does an $R^2$ score of 0.85 indicate?", 
             "It means that 85% of the total variance in the dependent target variable is explained by the features and the neural network model, with the remaining 15% attributed to unexplained residual variance.")
        ],
        "exam_takeaways": [
            "Output layer activation for regression is `linear` (or omitted).",
            "Loss is MSE or MAE, never cross-entropy.",
            "Evaluate using RMSE, MAE, or $R^2$ score."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=RCmiPBiA4qg"),
            ("Kaggle GRE Admission Notebook", "https://www.kaggle.com/campusx/gre-admission-prediction")
        ]
    },
    {
        "index": 14,
        "title": "Loss Functions in Deep Learning | Complete Taxonomy",
        "video_id": "gb5nm_3jBIo",
        "duration_str": "39m 10s",
        "core_intuition": r"""
The loss function is the mathematical compass of deep learning—it quantifies the discrepancy between model predictions $\hat{\mathbf{y}}$ and true targets $\mathbf{y}$.
The optimizer navigates the parameter landscape solely by following the negative gradient of this loss function.
Choosing the wrong loss function causes training to diverge, produce degenerate solutions, or optimize for the wrong operational objective.
This lecture presents the definitive taxonomy of loss functions for:
1. Regression: Mean Squared Error (L2), Mean Absolute Error (L1), and Huber Loss (Smooth L1).
2. Classification: Binary Cross-Entropy, Categorical Cross-Entropy, Sparse Categorical Cross-Entropy, and Hinge Loss.
""",
        "key_definitions": [
            ("Loss Function (Cost Function)", r"A mathematical function $\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}})$ that measures prediction error on a single instance (loss) or averaged over an entire dataset (cost)."),
            ("Huber Loss", "A hybrid loss function that behaves quadratically (like MSE) for small errors and linearly (like MAE) for large errors, combining differentiability with outlier robustness."),
            ("Kullback-Leibler (KL) Divergence", r"A measure of how one probability distribution $Q$ diverges from an expected reference probability distribution $P$: $D_{KL}(P \parallel Q) = \sum P(x) \log \frac{P(x)}{Q(x)}$. Cross-entropy is entropy plus KL divergence.")
        ],
        "mathematical_formulations": r"""
**Mathematical Definitions of Core Loss Functions:**

1. **Mean Squared Error (L2 Loss):**
   $$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

2. **Mean Absolute Error (L1 Loss):**
   $$\mathcal{L}_{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

3. **Huber Loss ($\delta$ threshold):**
   $$\mathcal{L}_{\delta}(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

4. **Binary Cross-Entropy (BCE):**
   $$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N [y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i)]$$

5. **Categorical Cross-Entropy (CCE):**
   $$\mathcal{L}_{CCE} = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{ik} \log(\hat{y}_{ik})$$
""",
        "architecture_and_algorithm": r"""
**Decision Tree for Selecting Loss Functions:**
```
Task Type?
├── Regression
│   ├── Outliers present and harmful?
│   │   ├── Yes -> Huber Loss (or MAE)
│   │   └── No  -> Mean Squared Error (MSE)
└── Classification
    ├── Binary (2 classes)
    │   └── Binary Cross-Entropy (with Sigmoid)
    ├── Multi-Class (Single label out of K)
    │   ├── Targets are one-hot? -> Categorical Cross-Entropy (with Softmax)
    │   └── Targets are integers? -> Sparse Categorical Cross-Entropy (with Softmax)
    └── Multi-Label (Multiple classes can be active)
        └── Binary Cross-Entropy per output neuron (with Sigmoid)
```
""",
        "code_snippet": r"""```python
import tensorflow as tf

# Using loss functions in Keras
mse_loss = tf.keras.losses.MeanSquaredError()
huber_loss = tf.keras.losses.Huber(delta=1.0)
bce_loss = tf.keras.losses.BinaryCrossentropy()
cce_loss = tf.keras.losses.CategoricalCrossentropy()
scce_loss = tf.keras.losses.SparseCategoricalCrossentropy()
```""",
        "exam_viva_qa": [
            ("Why is Huber loss preferred over MSE when training data contains severe noise and outliers?", 
             r"For large errors ($|y - \hat{y}| > \delta$), Huber loss transitions from a quadratic penalty to a linear penalty ($\delta |y - \hat{y}|$). Consequently, its gradient is bounded by $\pm \delta$ rather than growing proportionally to error magnitude, preventing exploding gradients from corrupting weights."),
            ("Show that Cross-Entropy minimization is equivalent to Maximum Likelihood Estimation.", 
             r"Under the Bernoulli assumption for binary targets, the likelihood is $\mathcal{L} = \prod \hat{y}_i^{y_i} (1 - \hat{y}_i)^{1 - y_i}$. Taking the negative natural log gives $-\ln \mathcal{L} = -\sum [y_i \ln \hat{y}_i + (1 - y_i) \ln(1 - \hat{y}_i)]$, which is the exact definition of Binary Cross-Entropy."),
            ("What loss function is used for Multi-Label image classification (e.g., an image containing both 'dog' and 'car')?", 
             "Binary Cross-Entropy with Sigmoid activation on each output neuron. Multi-label problems treat each class as an independent binary decision, whereas Softmax + Categorical Cross-Entropy forces probabilities to sum to 1.")
        ],
        "exam_takeaways": [
            "For Multi-label classification: Sigmoid + Binary Cross-Entropy.",
            "For Multi-class classification: Softmax + Categorical Cross-Entropy.",
            "Huber loss bridges the gap between MSE (smooth) and MAE (robust)."
        ],
        "resources_and_references": [
            ("Lecture Video", "https://www.youtube.com/watch?v=gb5nm_3jBIo")
        ]
    }
]
