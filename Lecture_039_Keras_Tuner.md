# Lecture 039: Keras Tuner | Hyperparameter Tuning a Neural Network

> **CampusX 100 Days of Deep Learning** | Video ID: `oYnyNLj8RMA` | Duration: 35m 45s
> **YouTube Link**: [Watch Video](https://www.youtube.com/watch?v=oYnyNLj8RMA) | **Transcript Status**: Available (hi, 8465 words)

---

## 1. Executive Summary & Core Intuition
While model weights are learned automatically via backpropagation, **Hyperparameters** (number of hidden layers, units per layer, activation functions, dropout rates, learning rates, optimizers) must be set by the engineer before training begins.

Manual trial-and-error is unscientific and inefficient.

This lecture introduces systematic automated hyperparameter optimization using **Keras Tuner**.

Four major search algorithms are evaluated:

1. **RandomSearch:** Samples random combinations from search space; remarkably competitive baseline.

2. **GridSearch:** Exhaustively evaluates Cartesian product; exponentially expensive ($\mathcal{O}(K^D)$).

3. **Hyperband:** Uses adaptive early-stopping and multi-armed bandit tournament brackets to rapidly discard poor configurations and allocate compute to top candidates.

4. **Bayesian Optimization:** Fits a Gaussian Process surrogate model of the objective function to intelligently explore promising configurations.


## 2. Key Definitions & Formal Terminology
- **Hyperparameter Tuning**: The process of systematically searching for the combination of architectural and training hyperparameters that maximizes validation performance.
- **Hyperband**: A bandit-based hyperparameter tuning algorithm that speeds up random search through aggressive early-stopping and successive halving resource allocation.
- **Surrogate Model (Bayesian Optimization)**: A probabilistic model (e.g., Gaussian Process) that approximates the true expensive objective function $\\mathcal{L}(\\theta)$ to predict high-yield search regions via Acquisition Functions (e.g., Expected Improvement).


## 3. Mathematical Formulations & Derivations
**Hyperband Resource Allocation:**

Hyperband balances the number of configurations $n$ with resource allocation $R$ (epochs per model) using **Successive Halving**:

Let maximum resource per model be $R$ and halving rate be $\eta_{hb} = 3$:

1. Train $n$ random configurations for $r = R / \eta_{hb}^s$ epochs.

2. Evaluate validation loss; keep only top $1 / \eta_{hb}$ (top 33%) models.

3. Train survivors for $\eta_{hb} \times r$ epochs.

4. Repeat until the single best model is trained for the full $R$ epochs.



Total compute is bounded by:

$$\text{Compute} \approx (s_{max} + 1) R$$

Hyperband evaluates $10\times$ more configurations than standard search for the same computational budget!


## 4. Architecture, Flowchart & Step-by-Step Algorithm
**Keras Tuner Model-Building Function Blueprint:**

```python

def build_model(hp):

    model = Sequential()

    # Tune number of layers

    for i in range(hp.Int('num_layers', 1, 4)):

        model.add(Dense(

            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),

            activation=hp.Choice(f'act_{i}', ['relu', 'tanh', 'elu'])

        ))

        if hp.Boolean(f'dropout_{i}'):

            model.add(Dropout(rate=hp.Float(f'drop_rate_{i}', 0.1, 0.5, step=0.1)))

    model.add(Dense(10, activation='softmax'))

    

    # Tune optimizer learning rate

    lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')

    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

```


## 5. Implementation Code Snippet
```python

import keras_tuner as kt



# Launching Hyperband Tuner

tuner = kt.Hyperband(

    build_model,

    objective='val_accuracy',

    max_epochs=30,

    factor=3,

    directory='tuner_results',

    project_name='mnist_tuning'

)



# Search

# tuner.search(X_train, y_train, epochs=30, validation_split=0.2)



# Retrieve best hyperparameters

# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# best_model = tuner.hypermodel.build(best_hps)

```


## 6. High-Yield Exam & Viva Questions with Answers
### Q1: Why is Random Search mathematically superior to Grid Search when tuning multiple hyperparameters?
**Answer**:
Bergstra & Bengio (2012) proved that in high dimensions, different hyperparameters have vastly different impacts on performance (some are critical like learning rate, others are minor like epsilon). Grid Search wastes evaluations testing identical values of important parameters against varying unimportant ones. Random Search evaluates unique values for all parameters on every trial, exploring the critical sub-manifold far more thoroughly.

### Q2: How does Hyperband's 'Successive Halving' work?
**Answer**:
It starts with a large pool of candidate models trained for just 1 or 2 epochs. After assessing initial trajectories, the bottom 66% of underperforming candidates are immediately terminated, and resources are doubled for the surviving top 33%. This tournament repeats, ensuring heavy compute is spent only on verified winners.

### Q3: Why should learning rates be sampled on a logarithmic scale rather than a linear scale?
**Answer**:
The impact of learning rate is multiplicative: the difference between $10^{-4}$ and $10^{-3}$ is an order of magnitude (10x), just like between $10^{-2}$ and $10^{-1}$. Sampling uniformly on $[0.0001, 0.1]$ would place 90% of samples above $0.01$, completely ignoring the lower orders of magnitude.

## 7. Crucial Exam Takeaways & Common Pitfalls
- Random Search is provably superior to Grid Search in high dimensions.
- Always sample learning rate on a LOGARITHMIC scale (`sampling='log'`).
- Hyperband uses successive halving to evaluate 10x more models efficiently.


## 8. References, Notebooks & Supplementary Materials
- [CampusX Video Link](https://www.youtube.com/watch?v=oYnyNLj8RMA)
- [Lecture Video](https://www.youtube.com/watch?v=oYnyNLj8RMA)

---
