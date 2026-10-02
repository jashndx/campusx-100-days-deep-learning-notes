# Lecture 053: Transfer Learning in Keras | Feature Extraction vs Fine Tuning

> **CampusX 100 Days of Deep Learning** | Video ID: `WWcgHjuKVqA` | Duration: 39m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=WWcgHjuKVqA) | **Transcript Status**: Available (en-US, 4909 words)

---

## 1. Executive Summary & Core Intuition
Transfer Learning is the superpower of modern applied deep learning: taking knowledge (features) learned by a model on a source task with millions of samples (e.g., ImageNet) and transferring it to a target task with limited samples (e.g., classifying 200 rare skin lesions).

Two core operational paradigms exist:

1. **Feature Extraction:** Freeze the entire pretrained convolutional base (`layer.trainable = False`). Remove the original 1000-class classification head, attach a new custom dense classification head, and train *only* the new head. Low compute, ultra-fast training, impossible to corrupt pretrained weights.

2. **Fine-Tuning:** Unfreeze the top few layers of the convolutional base (`layer.trainable = True`) and train both the custom head and top conv layers together using an **infinitesimally small learning rate** ($\eta \approx 10^{-5}$). This adapts high-level filters specifically to the target domain without destroying low-level primitives.


## 2. Key Definitions & Formal Terminology
- **Transfer Learning**: A machine learning method where a model developed for a task is reused as the starting point for a model on a second task.
- **Feature Extraction (TL)**: Using representations learned by a previous network to extract meaningful features from new samples, training only a new classifier head on top.
- **Fine-Tuning (TL)**: Unfreezing a few of the top layers of a frozen model base and jointly training both the newly added classifier layers and the top layers of the base model.
- **Catastrophic Forgetting**: The destructive phenomenon where training an un-frozen pretrained network with a standard large learning rate violently overwrites and destroys the valuable pretrained feature representations.


## 3. Mathematical Formulations & Derivations
**Decision Matrix for Transfer Learning Strategy:**



Let $N_{target}$ be the target dataset size, and $S_{similarity}$ be the semantic similarity between source (ImageNet) and target datasets:



$$\begin{array}{c|c|c}

& \text{High Dataset Similarity} & \text{Low Dataset Similarity} \\

\hline

\text{Small Data } (N < 10^3) & \textbf{Feature Extraction} & \textbf{Feature Extraction} \\

& (\text{Train only new head}) & (\text{Extract from early/mid layers}) \\

\hline

\text{Large Data } (N > 10^4) & \textbf{Fine-Tuning} & \textbf{Fine-Tuning / Train from Scratch} \\

& (\text{Unfreeze top conv blocks}) & (\text{Unfreeze entire network}) \\

\end{array}$$



**Learning Rate Differential in Fine-Tuning:**

$$\eta_{\text{fine-tune}} \le \frac{1}{10} \times \eta_{\text{scratch}} \approx 10^{-5} \text{ or } 10^{-6}$$

A tiny learning rate prevents disruptive gradient updates from destabilizing pretrained filter weights.


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Standard Two-Stage Transfer Learning Protocol:**

1. **Stage 1 (Feature Extraction):**

   - Load pretrained base (e.g., `VGG16(include_top=False)`).

   - Set `base_model.trainable = False`.

   - Attach GlobalAveragePooling2D + Dense head.

   - Train for 10 epochs with $\eta = 10^{-3}$ until head converges.

2. **Stage 2 (Fine-Tuning):**

   - Unfreeze the last 1-2 convolutional blocks: `base_model.trainable = True` (or selective layers).

   - Re-compile model with very low learning rate: $\eta = 10^{-5}$.

   - Train for another 15-20 epochs.


## 5. Implementation Code Snippet
```python

import tensorflow as tf

from tensorflow.keras import layers, models



# Stage 1: Feature Extraction

base_model = tf.keras.applications.VGG16(

    weights='imagenet',

    include_top=False,

    input_shape=(224, 224, 3)

)

base_model.trainable = False  # Freeze all base weights!



model = models.Sequential([

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dense(64, activation='relu'),

    layers.Dense(1, activation='sigmoid')

])



model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),

              loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(train_ds, epochs=10)



# Stage 2: Fine-Tuning

base_model.trainable = True

# Freeze all layers EXCEPT the last conv block (block5)

for layer in base_model.layers[:-4]:

    layer.trainable = False



# Must recompile with small learning rate!

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),

              loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(train_ds, epochs=15)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why must you train the custom classification head BEFORE unfreezing layers for fine-tuning?
**Answer**:
The newly added classification head starts with completely random weights. During the first few iterations, its prediction errors are massive, generating enormous error gradients. If the base layers are unfrozen, these massive random gradients will immediately propagate into the pretrained filters, triggering **Catastrophic Forgetting** and wrecking years of ImageNet training.

### Q2: Why does fine-tuning unfreeze only the top conv layers and NOT the earliest layers?
**Answer**:
The earliest layers (Conv block 1 & 2) learn generic universal low-level visual primitives (edges, lines, colors) that are useful across all vision tasks. High-level layers (Conv block 5) learn specific, complex semantic structures. Only the high-level layers need to be adapted to the nuances of the target dataset.

### Q3: What does `include_top=False` mean when loading a pretrained model in Keras?
**Answer**:
It instructs Keras to discard the original fully connected classification head (the Flatten + Dense(4096) + Dense(1000) layers trained on ImageNet) and return only the convolutional feature extraction backbone.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Always train the new top classifier head first with base frozen.
- Fine-tune with a 10x to 100x smaller learning rate ($10^{-5}$).
- Freeze early layers (edges are universal); adapt top layers.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=WWcgHjuKVqA)
- [Lecture Video](https://www.youtube.com/watch?v=WWcgHjuKVqA)
- [Colab Notebook](https://colab.research.google.com/drive/1VxoR4vMmZJAOCsDUnfezPuFQqHdKabcL?usp=sharing)

---
