# Lecture 075: Self-Attention Geometric Intuition | Subspaces & Projections

> **CampusX 100 Days of Deep Learning** | Video ID: `5ZgGuujZSbs` | Duration: 28m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=5ZgGuujZSbs) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
This lecture provides the visual and geometric intuition behind Self-Attention in high-dimensional vector spaces.

Geometrically:

1. Input word vectors exist as points/arrows in a $d$-dimensional semantic hyperspace (e.g., 512 dimensions).

2. The dot product $\mathbf{q} \cdot \mathbf{k}$ measures the **geometric cosine similarity** scaled by vector lengths:

   $$\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos(\theta)$$

   If two word concepts are semantically aligned in the subspace, their angle $\theta$ is small, $\cos(\theta) \to 1$, and their dot product is large.

3. The linear projection matrices $\mathbf{W}_Q, \mathbf{W}_K$ act as **subspace coordinate transforms**—they rotate and project raw embeddings into dedicated query and key hyperplanes where relevant task relationships become co-linear.

4. Multiplying by $\mathbf{V}$ performs a **convex combination (barycentric interpolation)**: the new token vector is a center-of-mass coordinate pulled toward the values of the words it attended to!


## 2. Key Definitions & Formal Terminology
- **Cosine Similarity**: A measure of directional similarity between two non-zero vectors in an inner product space: $\\cos(\\theta) = \\frac{\\mathbf{A} \\cdot \\mathbf{B}}{\\|\\mathbf{A}\\| \\|\\mathbf{B}\\|}$.
- **Convex Combination**: A linear combination of points where all coefficients are non-negative and sum strictly to $1$: $\\sum \\alpha_i \\mathbf{v}_i$ where $\\alpha_i \\ge 0, \\sum \\alpha_i = 1$.
- **Semantic Subspace**: A lower-dimensional vector space spanned by learned projection matrices where specific linguistic relationships (e.g., subject-verb agreement, gender) are isolated.


## 3. Mathematical Formulations & Derivations
**Geometric Interpretation of Attention as a Barycentric Hull:**



Because $\sum_{j=1}^N A_{ij} = 1.0$ and $A_{ij} \ge 0$ for all $j$:

$$\mathbf{z}_i = \sum_{j=1}^N A_{ij} \mathbf{v}_j$$



Geometrically, the output vector $\mathbf{z}_i$ is strictly constrained to lie inside the **Convex Hull** formed by the set of all Value vectors $\{\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_N\}$:



$$\mathbf{z}_i \in \text{Conv}(\{\mathbf{v}_1, \dots, \mathbf{v}_N\})$$



If word $i$ pays 100% attention to word $j$, $\mathbf{z}_i$ jumps to vertex $\mathbf{v}_j$. 

If it balances attention between two words equally, $\mathbf{z}_i$ settles exactly at their midpoint!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
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


## 5. Implementation Code Snippet
```python

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

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is the dot product used as a similarity measure in self-attention rather than Euclidean distance?
**Answer**:
Dot product combines both angle (directional alignment) and magnitude (importance/salience). Furthermore, computing dot products across an entire sequence is a single matrix multiplication $\mathbf{Q}\mathbf{K}^T$, which is orders of magnitude faster on GPUs than computing pairwise Euclidean distances $\|q_i - k_j\|_2$.

### Q2: What does it mean geometrically that $\mathbf{z}_i$ lies in the convex hull of the Value vectors?
**Answer**:
It means self-attention cannot synthesize entirely new vector magnitudes or directions out of thin air; it can only re-weight, interpolate, and blend existing informational values that were already present in the sequence.

### Q3: How do learned projection matrices $\mathbf{W}_Q, \mathbf{W}_K$ enable attention to identify non-obvious relationships?
**Answer**:
Two words might have orthogonal or distant raw embeddings. The linear projection matrices $\mathbf{W}_Q$ and $\mathbf{W}_K$ project those vectors into a transformed latent subspace where their projected coordinates become aligned (high dot product), allowing the model to learn task-specific contextual dependencies.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Dot product measures directional similarity: $\mathbf{q} \cdot \mathbf{k} = \|\mathbf{q}\| \|\mathbf{k}\| \cos\theta$.
- Output $\mathbf{z}_i$ is a convex combination lying inside the convex hull of Value vectors.
- Learned matrices project tokens into specialized functional subspaces.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=5ZgGuujZSbs)
- [Lecture Video](https://www.youtube.com/watch?v=5ZgGuujZSbs)
- [Official 100 Days of Deep Learning Repo](https://github.com/campusx-official/100-days-of-deep-learning)

---
