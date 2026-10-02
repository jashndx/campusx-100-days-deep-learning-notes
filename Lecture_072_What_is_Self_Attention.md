# Lecture 072: What is Self-Attention? | The Query, Key, Value Metaphor

> **CampusX 100 Days of Deep Learning** | Video ID: `XnGGmvpDLA0` | Duration: 32m 40s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=XnGGmvpDLA0) | **Transcript Status**: Available via official notes & syllabus outline

---

## 1. Executive Summary & Core Intuition
What is Self-Attention conceptually, and how does a sentence attend to itself?

Consider the sentence: "The **animal** didn't cross the street because **it** was too tired."

What does the pronoun "**it**" refer to? The animal or the street?

To a human, "tired" implies an animate creature, so "it" refers to the animal.

Self-Attention allows the word "it" to look across all other words in the sentence, compute affinity scores, and blend the representation of "animal" into the representation of "it"!

To compute this interaction dynamically, Vaswani et al. borrowed the **Query, Key, Value (Q, K, V)** retrieval metaphor from database and search engine systems:

- **Query ($\mathbf{Q}$):** 'What am I looking for?' (The current word seeking context).

- **Key ($\mathbf{K}$):** 'What do I contain?' (Every word's identity tag).

- **Value ($\mathbf{V}$):** 'What is my actual content/meaning?' (The information delivered if a match is made).


## 2. Key Definitions & Formal Terminology
- **Query ($\mathbf{Q}$)**: A vector representing the current token's search intent when probing the sequence for contextual relationships.
- **Key ($\mathbf{K}$)**: A vector representing a token's indexable label that is compared against incoming queries via dot products.
- **Value ($\mathbf{V}$)**: A vector representing the actual informative content that is aggregated into the output representation once query and key match.
- **Contextualized Representation**: A dynamic token representation that incorporates weighted semantic information from relevant surrounding tokens.


## 3. Mathematical Formulations & Derivations
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


## 4. Architecture, Flowchart & Step-by-Step Algorithm
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


## 5. Implementation Code Snippet
```python

import numpy as np



# Conceptual Self-Attention for single word

# Words: ["The", "animal", "crossed", "it"]

# If query is "it", its dot product with key("animal") is high!

query_it = np.array([0.1, 0.9, 0.2])

key_animal = np.array([0.1, 0.8, 0.3])

key_street = np.array([0.8, 0.1, 0.1])



score_animal = np.dot(query_it, key_animal) # 0.01 + 0.72 + 0.06 = 0.79 (High match!)

score_street = np.dot(query_it, key_street) # 0.08 + 0.09 + 0.02 = 0.19 (Low match!)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Explain the Query, Key, Value metaphor using YouTube search as an analogy.
**Answer**:
When you search for 'deep learning tutorial' in YouTube's search bar, that text is your **Query**. YouTube compares your query against the titles, tags, and metadata (**Keys**) of millions of cataloged videos. The similarity match between query and keys determines rank/weights. Finally, YouTube delivers the actual video content (**Values**) of the matching results.

### Q2: Why does every word have THREE separate vectors ($Q, K, V$) instead of just using its original embedding vector?
**Answer**:
A word serves different roles in different interactions: when seeking context, it acts as a Query; when providing context to another word, it acts as a Key; when delivering semantic content, it acts as a Value. Having separate learnable projection matrices ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$) decouples these three roles, providing vastly greater expressive capacity.

### Q3: What does the self-attention output $\mathbf{z}_i$ for a word represent?
**Answer**:
It represents the updated, contextualized embedding of word $i$. It retains the core identity of the original word while infusing relevant contextual nuances from all other words in the sentence that the word attended to.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Query = What I am looking for; Key = What I contain; Value = What I deliver.
- Soft database retrieval: $\text{Output} = \sum \text{Softmax}(Q \cdot K^T) V$.
- Transforms static word embeddings into contextualized representations.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=XnGGmvpDLA0)
- [Lecture Video](https://www.youtube.com/watch?v=XnGGmvpDLA0)

---
