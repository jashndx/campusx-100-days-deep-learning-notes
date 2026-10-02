# 🧠 CampusX 100 Days of Deep Learning — Comprehensive Notes & Companion Guide

[![Course Playlist](https://img.shields.io/badge/YouTube-100%20Days%20of%20Deep%20Learning-red?logo=youtube)](https://www.youtube.com/watch?v=2dH_qjc9mFg&list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)
[![Lectures Covered](https://img.shields.io/badge/Lectures%20Covered-84%20%2F%2084%20(100%25)-brightgreen)]()
[![Code Snippets Verified](https://img.shields.io/badge/AST%20Verified%20Code-170%20Snippets-blue)]()
[![Official Notebooks](https://img.shields.io/badge/Colab%20Notebooks-31%20Archived-orange)]()
[![Seminal Papers](https://img.shields.io/badge/Research%20Papers-6%20ArXiv%20PDFs-purple)]()
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Attribution & Acknowledgement**:  
> This open-source study repository is an independent companion to the acclaimed **[100 Days of Deep Learning](https://www.youtube.com/playlist?list=PLKnIA16_RmvYuZauWaPlRTC54KxSNLtNn)** playlist taught by **[Nitish Singh](https://www.linkedin.com/in/nitish-singh-03412789/)** on **[CampusX](https://www.youtube.com/@CampusX-official)**.  
> All lecture content, syllabus structure, and foundational pedagogy belong to Nitish Singh and CampusX. This repository was created by students to document the deep mathematical rigor, conceptual derivations, and practical implementations from the course.

---

## 📌 Why This Repository Exists

CampusX's *100 Days of Deep Learning* is one of the most comprehensive Hindi/English deep learning courses available, breaking down intimidating mathematics (backpropagation computational graphs, LSTM gates, Multi-Head Attention, and Transformer inference) into clear, first-principles intuition.

However, watching 84+ hours of video during rapid revision, concept lookup, or exam preparation is challenging. This repository provides:
1. **Pristine Mathematical Rigor**: Full LaTeX derivations for Perceptron Loss, Backpropagation chain rules, Batch Normalization, Attention score matrices, and Transformer KV Caching.
2. **Master Revision Compendium**: A single 523 KB searchable textbook ([`CONSOLIDATED_MASTER_EXAM_NOTES.md`](./CONSOLIDATED_MASTER_EXAM_NOTES.md)) combining all 84 lectures.
3. **Curated Code & Notebooks**: 31 official Google Colab `.ipynb` notebooks archived directly from video descriptions into [`slides/notebooks_and_code/`](./slides/notebooks_and_code/).
4. **Seminal Research Papers**: 6 seminal ArXiv research papers referenced throughout the series in [`slides/research_papers_and_readings/`](./slides/research_papers_and_readings/).
5. **Interview & Exam Viva Q&A**: Over 300+ conceptual and viva-style questions with verified model answers across all modules.

---

## 🗂️ Repository Structure

```text
deep_learning_100_exam_notes/
│
├── CONSOLIDATED_MASTER_EXAM_NOTES.md       # Master 523 KB single-file revision compendium
├── Lecture_001_What_is_Deep_Learning.md    # Detailed lecture note (Lec 01)
├── ...
├── Lecture_084_Transformer_Inference...md  # Detailed lecture note (Lec 84)
│
├── slides/
│   ├── notebooks_and_code/                 # 31 official Colab .ipynb notebooks from CampusX
│   │   ├── day3/
│   │   ├── day4/
│   │   ├── day5 - Perceptron Loss Function/
│   │   ├── Lecture_016_Backpropagation...ipynb
│   │   ├── Lecture_049_Cat_Vs_Dog_CNN...ipynb
│   │   ├── Lecture_063_LSTM_Next_Word...ipynb
│   │   ├── Lecture_077_Multi_Head_Attention...ipynb
│   │   └── ...
│   ├── research_papers_and_readings/       # 6 Seminal ArXiv research papers (PDF)
│   │   ├── Attention_Is_All_You_Need_Vaswani2017.pdf
│   │   ├── Deep_Residual_Learning_for_Image_Recognition_He2015.pdf
│   │   ├── Adam_Method_For_Stochastic_Optimization_Kingma2014.pdf
│   │   ├── Batch_Normalization_Accelerating_Deep_Network_Ioffe2015.pdf
│   │   ├── Layer_Normalization_Ba2016.pdf
│   │   └── Bahdanau_Attention_Neural_Machine_Translation_2014.pdf
│   ├── Dropout_A_Simple_Way_to_Prevent_Overfitting_Srivastava2014.pdf
│   └── Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf
│
├── test_notes_integrity.py                 # Automated AST parser and formula verification suite
└── README.md                               # Course navigation and curriculum guide
```

---

## 🗺️ Syllabus & Lecture Index

### Module 1: Foundations of Deep Learning & Perceptrons (Lec 01 – 11)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook |
|---|---|---|---|
| [**Lec 01**](./Lecture_001_What_is_Deep_Learning.md) | What is Deep Learning? | AI vs ML vs DL, representation learning, hierarchy of features | — |
| [**Lec 02**](./Lecture_002_Biological_Neuron_vs_Artificial_Neuron.md) | Biological vs Artificial Neuron | Dendrites, soma, axon, synapses to inputs, weights, bias | — |
| [**Lec 03**](./Lecture_003_Perceptron_in_Deep_Learning.md) | Perceptron Architecture | Heaviside step function, linear separability, decision boundary | [day3/demo](./slides/notebooks_and_code/day3) |
| [**Lec 04**](./Lecture_004_Perceptron_Trick.md) | The Perceptron Trick | Geometric intuition: $w_{new} = w_{old} + \eta (y_i - \hat{y}_i) x_i$ | [day4/demo](./slides/notebooks_and_code/day4) |
| [**Lec 05**](./Lecture_005_Perceptron_Loss_Function.md) | Perceptron Loss Formulation | Non-differentiability of step function, hinge-style loss | [day5/demo](./slides/notebooks_and_code/day5%20-%20Perceptron%20Loss%20Function) |
| [**Lec 06**](./Lecture_006_Gradient_Descent_in_Perceptron.md) | Gradient Descent Derivation | Partial derivatives of loss w.r.t weights and bias | — |
| [**Lec 07**](./Lecture_007_Problem_with_Perceptron.md) | Limitations of Perceptron | The XOR problem (Minsky & Papert 1969), linear bottleneck | [Lec 07 Colab](./slides/notebooks_and_code/Lecture_007_Problem_with_Perceptron.ipynb) |
| [**Lec 08**](./Lecture_008_Loss_Functions_in_Machine_Learning.md) | Loss Functions Overview | Regression (MSE, MAE, Huber) vs Classification (Binary Cross-Entropy) | — |
| [**Lec 09**](./Lecture_009_Activation_Functions_Sigmoid_Tanh_ReLU.md) | Classic Activation Functions | Sigmoid, Tanh, ReLU, vanishing gradient at tails | — |
| [**Lec 10**](./Lecture_010_Activation_Functions_Part_2_Leaky_ReLU_ELU_Softmax.md) | Modern Activation Functions | Leaky ReLU, Parametric ReLU, ELU, Softmax derivation | — |
| [**Lec 11**](./Lecture_011_Softmax_Activation_Function.md) | Softmax & Multi-Class Loss | Probability distribution output, Categorical Cross-Entropy | — |

---

### Module 2: Multilayer Perceptrons & Backpropagation (Lec 12 – 19)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook |
|---|---|---|---|
| [**Lec 12**](./Lecture_012_Handwritten_Digit_Classification_MNIST.md) | MNIST Digit Classification | First deep neural network in Keras/TensorFlow | [Lec 12 Colab](./slides/notebooks_and_code/Lecture_012_Handwritten_Digit_Classification_MNIST.ipynb) |
| [**Lec 13**](./Lecture_013_What_is_a_Multilayer_Perceptron_MLP.md) | MLP Architecture & Theory | Hidden layers, universal approximation theorem | — |
| [**Lec 14**](./Lecture_014_Forward_Propagation_in_Neural_Networks.md) | Forward Propagation | Matrix notation: $\mathbf{Z}^{[l]} = \mathbf{W}^{[l]}\mathbf{A}^{[l-1]} + \mathbf{b}^{[l]}$ | — |
| [**Lec 15**](./Lecture_015_Backpropagation_Part_1.md) | Backprop Foundations | Computational graphs, reverse-mode automatic differentiation | — |
| [**Lec 16**](./Lecture_016_Backpropagation_Part_2.md) | Full Backprop Mathematical Derivation | Complete multivariate chain rule derivations | [Lec 16 Colab](./slides/notebooks_and_code/Lecture_016_Backpropagation_Part_2.ipynb) |
| [**Lec 17**](./Lecture_017_Backpropagation_Intuition.md) | Backpropagation Intuition | Credit assignment problem, error redistribution | — |
| [**Lec 18**](./Lecture_018_Vanishing_Gradient_Problem.md) | Vanishing Gradients | Derivative decay across deep layers: $\prod \sigma'(z) \to 0$ | [Lec 18 Colab](./slides/notebooks_and_code/Lecture_018_Vanishing_Gradient_Problem.ipynb) |
| [**Lec 19**](./Lecture_019_Exploding_Gradient_Problem.md) | Exploding Gradients & Clipping | Eigenvalues of weight matrices, gradient clipping by norm/value | — |

---

### Module 3: Optimization & Regularization (Lec 20 – 35)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook / Paper |
|---|---|---|---|
| [**Lec 20**](./Lecture_020_Gradient_Descent_in_Neural_Networks.md) | Batch, Mini-Batch & SGD | Computational complexity, memory trade-offs, noisy trajectories | [Lec 20 Colab](./slides/notebooks_and_code/Lecture_020_Gradient_Descent_in_Neural_Networks.ipynb) |
| [**Lec 21**](./Lecture_021_How_to_Improve_the_Performance_of_Neural_Networks.md) | Diagnostic Strategies | High bias vs high variance, learning curve diagnostics | [Lec 21 Colab](./slides/notebooks_and_code/Lecture_021_How_to_Improve_the_Performance_of_Neural_Networks.ipynb) |
| [**Lec 22**](./Lecture_022_Early_Stopping_In_Neural_Networks.md) | Early Stopping Regularization | Validation loss patience, restore best weights mechanism | [Lec 22 Colab](./slides/notebooks_and_code/Lecture_022_Early_Stopping_In_Neural_Networks.ipynb) |
| [**Lec 23**](./Lecture_023_Data_Scaling_in_Neural_Networks.md) | Feature Scaling | Standardization vs MinMax, spherical loss surfaces | [Lec 23 Colab](./slides/notebooks_and_code/Lecture_023_Data_Scaling_in_Neural_Networks.ipynb) |
| [**Lec 24**](./Lecture_024_Why_Do_Neural_Networks_Overfit.md) | Overfitting Dynamics | Parameter capacity vs sample size, memorization regime | — |
| [**Lec 25**](./Lecture_025_Dropout_Layers_in_ANN.md) | Dropout Regularization | Bernoulli mask $r_j \sim \text{Bernoulli}(p)$, inverted dropout | [Dropout Paper PDF](./slides/Dropout_A_Simple_Way_to_Prevent_Overfitting_Srivastava2014.pdf) |
| [**Lec 26**](./Lecture_026_Regularization_in_Deep_Learning_L1_and_L2.md) | L1 / L2 Weight Decay | Lasso ($L_1$) sparsity vs Ridge ($L_2$) Frobenius norm penalty | [Lec 26 Colab](./slides/notebooks_and_code/Lecture_026_Regularization_in_Deep_Learning_L1_and_L2.ipynb) |
| [**Lec 27**](./Lecture_027_Weight_Initialization_Techniques_in_Deep_Learning.md) | Weight Initialization Foundations | Zero init symmetry problem, random normal variance explosion | — |
| [**Lec 28**](./Lecture_028_Vanishing_and_Exploding_Gradients_Due_to_Weights.md) | Variance Analysis of Initializations | Variance preservation across forward and backward passes | — |
| [**Lec 29**](./Lecture_029_Weight_Initialization_Techniques_Zero_Random_Normal.md) | Practical Initializations | Uniform vs Normal distributions, scale tuning | [Lec 29 Colab](./slides/notebooks_and_code/Lecture_029_Weight_Initialization_Techniques_Zero_Random_Normal.ipynb) |
| [**Lec 30**](./Lecture_030_XavierGlorot_And_He_Weight_Initialization.md) | Xavier/Glorot & He Initialization | $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$ (Glorot), $\text{Var}(W) = \frac{2}{n_{in}}$ (He) | [Lec 30 Colab](./slides/notebooks_and_code/Lecture_030_XavierGlorot_And_He_Weight_Initialization.ipynb) |
| [**Lec 31**](./Lecture_031_Batch_Normalization_in_Deep_Learning.md) | Batch Normalization | Internal covariate shift, scale & shift parameters $\gamma, \beta$ | [BatchNorm Paper PDF](./slides/research_papers_and_readings/Batch_Normalization_Accelerating_Deep_Network_Ioffe2015.pdf) |
| [**Lec 32**](./Lecture_032_Optimizers_in_Deep_Learning.md) | Optimizers Overview | Ravines, saddle points, pathological curvature challenges | — |
| [**Lec 33**](./Lecture_033_Momentum_and_Nesterov_Accelerated_Gradient_NAG.md) | Momentum & NAG | Exponential moving average of gradients, lookahead gradient step | — |
| [**Lec 34**](./Lecture_034_AdaGrad_and_RMSprop_Optimizers.md) | Adaptive Learning Rates | AdaGrad accumulated squared gradients, RMSprop leaky average | — |
| [**Lec 35**](./Lecture_035_Adam_Optimizer.md) | Adam Optimizer | Bias-corrected 1st ($m_t$) and 2nd ($v_t$) moment estimation | [Adam Paper PDF](./slides/research_papers_and_readings/Adam_Method_For_Stochastic_Optimization_Kingma2014.pdf) |

---

### Module 4: Convolutional Neural Networks (CNNs) (Lec 36 – 54)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook / Paper |
|---|---|---|---|
| [**Lec 36**](./Lecture_036_Introduction_to_Computer_Vision_and_CNNs.md) | Intro to Computer Vision | Why MLPs fail on high-res images, parameter explosion | — |
| [**Lec 37**](./Lecture_037_Convolution_Operation.md) | The 2D Convolution Operation | Kernel sliding, element-wise dot product, feature maps | — |
| [**Lec 38**](./Lecture_038_Filters_and_Feature_Detectors_in_CNNs.md) | Filters & Edge Detection | Sobel, Prewitt, Laplacian edge detection kernels | — |
| [**Lec 39**](./Lecture_039_Convolution_Over_RGB_Images.md) | 3D Convolution | Multi-channel input tensor $H \times W \times C$, 3D filter kernels | — |
| [**Lec 40**](./Lecture_040_Convolution_with_Multiple_Filters.md) | Multi-Filter Convolutions | Stacking feature maps into output depth $D_{out}$ | — |
| [**Lec 41**](./Lecture_041_Padding_in_CNNs.md) | Valid vs Same Padding | Output spatial dimension: $\lfloor\frac{N + 2P - F}{S} + 1\rfloor$ | — |
| [**Lec 42**](./Lecture_042_Strides_in_CNNs.md) | Stride Mechanics | Subsampling during convolution, computational reduction | — |
| [**Lec 43**](./Lecture_043_Padding__Strides_in_CNN.md) | Combined Padding & Strides | Keras implementation and dimension calculus | [Lec 43 Colab](./slides/notebooks_and_code/Lecture_043_Padding__Strides_in_CNN.ipynb) |
| [**Lec 44**](./Lecture_044_Pooling_Layer_in_CNN__Max_Average_Global.md) | Pooling Layers | Max pooling, average pooling, translation invariance | [Lec 44 Colab](./slides/notebooks_and_code/Lecture_044_Pooling_Layer_in_CNN__Max_Average_Global.ipynb) |
| [**Lec 45**](./Lecture_045_CNN_Architecture_End_to_End.md) | End-to-End CNN Pipeline | CONV $\to$ RELU $\to$ POOL $\to$ FLATTEN $\to$ DENSE | — |
| [**Lec 46**](./Lecture_046_LeNet5_CNN_Architecture.md) | LeNet-5 Architecture | Yann LeCun's 1998 pioneering digit recognition network | — |
| [**Lec 47**](./Lecture_047_AlexNet_CNN_Architecture.md) | AlexNet Architecture | 2012 ImageNet breakthrough: ReLU, Dropout, GPU training | — |
| [**Lec 48**](./Lecture_048_VGGNet_CNN_Architecture.md) | VGG-16 / VGG-19 Architecture | Uniform $3 \times 3$ convolutions, deep representation | — |
| [**Lec 49**](./Lecture_049_Cat_Vs_Dog_Image_Classification_using_CNN.md) | Binary Classification Project | Dogs vs Cats image classifier with data augmentation | [Lec 49 Colab](./slides/notebooks_and_code/Lecture_049_Cat_Vs_Dog_Image_Classification_using_CNN.ipynb) |
| [**Lec 50**](./Lecture_050_Data_Augmentation_in_CNNs.md) | Data Augmentation | Rotation, zoom, shear, horizontal flip regularization | — |
| [**Lec 51**](./Lecture_051_GoogLeNet_Inception_Network.md) | GoogLeNet / Inception v1 | Multi-scale parallel kernels, $1 \times 1$ bottleneck convolutions | — |
| [**Lec 52**](./Lecture_052_What_does_a_CNN_see__Visualizing_Filters.md) | CNN Feature Visualization | Filter activation maximization, Grad-CAM concepts | [Lec 52 Colab](./slides/notebooks_and_code/Lecture_052_What_does_a_CNN_see__Visualizing_Filters.ipynb) |
| [**Lec 53**](./Lecture_053_What_is_Transfer_Learning.md) | Transfer Learning & Fine-Tuning | Feature extractor vs fine-tuning, catastrophic forgetting | [Lec 53 Colab](./slides/notebooks_and_code/Lecture_053_What_is_Transfer_Learning.ipynb) |
| [**Lec 54**](./Lecture_054_Keras_Functional_Model_API.md) | Keras Functional API | Multi-input, multi-output, residual skip graph construction | [Lec 54 Colab](./slides/notebooks_and_code/Lecture_054_Keras_Functional_Model_API.ipynb) |

---

### Module 5: Sequential Models (RNN, LSTM, GRU) (Lec 55 – 68)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook |
|---|---|---|---|
| [**Lec 55**](./Lecture_055_Introduction_to_Sequence_Models_and_RNNs.md) | Sequence Modeling Foundations | Text, speech, time-series, tokenization, vocabulary | — |
| [**Lec 56**](./Lecture_056_Recurrent_Neural_Network_Architecture.md) | Vanilla RNN Architecture | Hidden state update: $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$ | — |
| [**Lec 57**](./Lecture_057_RNN_Sentiment_Analysis_Project.md) | IMDB Sentiment Analysis | Text classification with SimpleRNN in Keras | [Lec 57 Colab](./slides/notebooks_and_code/Lecture_057_RNN_Sentiment_Analysis_Project.ipynb) |
| [**Lec 58**](./Lecture_058_Backpropagation_Through_Time_BPTT.md) | BPTT Derivation | Unfolding computational graphs across timesteps | — |
| [**Lec 59**](./Lecture_059_Vanishing_and_Exploding_Gradients_in_RNNs.md) | The Long-Term Memory Bottleneck | Repeated matrix multiplication by $W_{hh}^T$, gradient vanishing | — |
| [**Lec 60**](./Lecture_060_Introduction_to_LSTM.md) | Introduction to LSTM | Cell state as the constant error carousel (Hochreiter 1997) | — |
| [**Lec 61**](./Lecture_061_LSTM_Architecture_Walkthrough.md) | Full LSTM Architecture | Forget gate ($f_t$), Input gate ($i_t$), Candidate ($\tilde{C}_t$), Output ($o_t$) | — |
| [**Lec 62**](./Lecture_062_Mathematical_Formulation_of_LSTM.md) | Mathematical Equations of LSTM | Complete vectorized LSTM state transition equations | — |
| [**Lec 63**](./Lecture_063_LSTM__Part_3__Next_Word_Prediction.md) | Next-Word Prediction with LSTM | Language modeling with character/word token sequences | [Lec 63 Colab](./slides/notebooks_and_code/Lecture_063_LSTM__Part_3__Next_Word_Prediction.ipynb) |
| [**Lec 64**](./Lecture_064_Gated_Recurrent_Unit_GRU.md) | Gated Recurrent Unit (GRU) | Cho et al. 2014: Reset gate ($r_t$), Update gate ($z_t$) | — |
| [**Lec 65**](./Lecture_065_Deep_RNNs__Stacked_RNNs.md) | Stacked Deep RNNs & LSTMs | Hierarchical temporal abstraction, Keras `return_sequences=True` | [Lec 65 Colab](./slides/notebooks_and_code/Lecture_065_Deep_RNNs__Stacked_RNNs.ipynb) |
| [**Lec 66**](./Lecture_066_Bidirectional_RNNs_and_LSTMs.md) | Bidirectional RNNs | Forward state $\vec{h}_t$ and backward state $\overleftarrow{h}_t$ concatenation | — |
| [**Lec 67**](./Lecture_067_Seq2Seq_Architecture.md) | Seq2Seq Architecture | Encoder-Decoder framework for Machine Translation | — |
| [**Lec 68**](./Lecture_068_The_Information_Bottleneck_Problem_in_Seq2Seq.md) | Seq2Seq Bottleneck Problem | Fixed-size context vector $c$ failing on long sentences | — |

---

### Module 6: Attention Mechanisms & Transformers (Lec 69 – 84)
| Lecture | Topic Title | Core Concepts & Formulas | Colab Notebook / Paper |
|---|---|---|---|
| [**Lec 69**](./Lecture_069_Introduction_to_Attention_Mechanism.md) | The Attention Intuition | Dynamic context vector $c_i = \sum \alpha_{ij} h_j$, alignment scores | [Bahdanau Attention Paper PDF](./slides/research_papers_and_readings/Bahdanau_Attention_Neural_Machine_Translation_2014.pdf) |
| [**Lec 70**](./Lecture_070_Bahdanau_Additive_Attention.md) | Bahdanau Additive Attention | Alignment model: $e_{ij} = v_a^T \tanh(W_a s_{i-1} + U_a h_j)$ | — |
| [**Lec 71**](./Lecture_071_Luong_Multiplicative_Attention.md) | Luong Multiplicative Attention | Dot, General, Concat attention score functions | — |
| [**Lec 72**](./Lecture_072_Self_Attention_Mechanism.md) | Self-Attention Mechanism | Intra-sequence dependency modeling without recurrence | — |
| [**Lec 73**](./Lecture_073_Scaled_Dot_Product_Attention.md) | Scaled Dot-Product Attention | $\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | — |
| [**Lec 74**](./Lecture_074_Why_Scale_in_Scaled_Dot_Product_Attention.md) | Why Divide by $\sqrt{d_k}$? | Large dot-products push Softmax into tiny gradient regions | — |
| [**Lec 75**](./Lecture_075_Query_Key_Value_Matrices_in_Self_Attention.md) | Query, Key, Value Projections | Linear transformation matrices $W^Q, W^K, W^V$ | — |
| [**Lec 76**](./Lecture_076_Multi_Head_Attention_Intuition.md) | Multi-Head Attention Intuition | Multiple representation subspaces (syntax, grammar, semantics) | [Attention Is All You Need PDF](./slides/research_papers_and_readings/Attention_Is_All_You_Need_Vaswani2017.pdf) |
| [**Lec 77**](./Lecture_077_What_is_Multi-head_Attention.md) | Multi-Head Attention Math | $\text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$ formulation | [Lec 77 Colab](./slides/notebooks_and_code/Lecture_077_What_is_Multi-head_Attention.ipynb) |
| [**Lec 78**](./Lecture_078_Positional_Encoding_in_Transformers.md) | Positional Encoding | Sinusoidal encodings: $\sin / \cos(pos / 10000^{2i/d_{model}})$ | [Transformers Course Notes PDF](./slides/Transformers_Complete_Course_Notes_Lectures_78_to_84.pdf) |
| [**Lec 79**](./Lecture_079_Layer_Normalization_vs_Batch_Normalization.md) | LayerNorm vs BatchNorm | Normalizing across feature dimensions vs across mini-batch | [LayerNorm Paper PDF](./slides/research_papers_and_readings/Layer_Normalization_Ba2016.pdf) |
| [**Lec 80**](./Lecture_080_Transformer_Encoder_Architecture.md) | Complete Transformer Encoder | Multi-Head Self-Attention + Residual + LayerNorm + FFN | — |
| [**Lec 81**](./Lecture_081_Masked_Multi_Head_Attention_in_Decoder.md) | Masked Self-Attention | Lower-triangular causal mask to prevent lookahead cheating | — |
| [**Lec 82**](./Lecture_082_Cross_Attention_in_Transformer_Decoder.md) | Cross-Attention Mechanics | Decoder queries ($Q$) attending to Encoder keys ($K$) & values ($V$) | — |
| [**Lec 83**](./Lecture_083_Complete_Transformer_Architecture_End_to_End.md) | End-to-End Transformer Pipeline | Joint training, label smoothing, learning rate warmup | — |
| [**Lec 84**](./Lecture_084_Transformer_Inference_and_KV_Cache.md) | Transformer Inference & KV Caching | Autoregressive decoding, eliminating $O(N^2)$ redundant compute | — |

---

## ⚡ Quick Start: How to Study

### 1. Unified Search / Cramming (Fastest)
Open [`CONSOLIDATED_MASTER_EXAM_NOTES.md`](./CONSOLIDATED_MASTER_EXAM_NOTES.md) in your editor. Press `Ctrl + F` to search for any formula, layer name, or algorithm.

### 2. Topic-by-Topic Study
Click on any lecture link in the index table above to read the dedicated note, review the step-by-step mathematical derivations, and inspect the code examples.

### 3. Interview / Exam Self-Testing
Every lecture note ends with an **Exam / Viva Questions** section. Test yourself on these questions before checking the model answers.

---

## 🧪 Verification & Quality Standards

All notes and assets in this repository are verified using automated testing scripts:
- **0 Corrupted Characters**: Formulas encoded in raw LaTeX strings.
- **100% Valid Python Syntax**: All 170 code blocks verified via Python's Abstract Syntax Tree (`ast.parse`).
- **Archive Integrity**: 31 official Google Colab notebooks and 6 seminal ArXiv research papers verified on disk.

To re-run the verification suite locally:
```bash
python test_notes_integrity.py
```

---

## 🤝 Contributing & Community Feedback

We welcome contributions, typo fixes, and community extensions:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feat/enhance-lecture-XX`).
3. Commit your changes and submit a Pull Request.

---

## 📜 License & Credit

- **Pedagogy & Course Materials**: © [CampusX](https://www.youtube.com/@CampusX-official) & [Nitish Singh](https://www.linkedin.com/in/nitish-singh-03412789/).
- **Study Notes & Markdown Synthesis**: Licensed under [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Free for personal study, student groups, and non-commercial educational use.
