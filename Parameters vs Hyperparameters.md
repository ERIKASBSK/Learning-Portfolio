# Note 1 — Parameters vs Hyperparameters
In a neural network, **parameters** are learned from data:

- weights: $$W$$  
- biases: $$b$$  

**Hyperparameters** are values i choose before/during training to control learning:

- learning rate: $$\alpha$$  
- number of iterations / epochs  
- number of layers: $$L$$  
- hidden units per layer: $$n^{[1]}, n^{[2]}, \dots$$  
- activation functions (ReLU / tanh / sigmoid)  
- (later) batch size, momentum, regularization strength, etc.

---

# Note 2 — Why they are called “hyper”
Hyperparameters don’t directly become the model’s knowledge.

Instead, they **control how the model learns**, which determines the final $$W$$ and $$b$$.

A basic gradient descent update:

$$
W := W - \alpha \frac{\partial J}{\partial W}
$$

$$
b := b - \alpha \frac{\partial J}{\partial b}
$$

So changing $$\alpha$$ can completely change the training outcome.

---

# Note 3 — Learning rate $$\alpha$$ (most sensitive one)
I treat $$\alpha$$ like the step size:

- too small → training is very slow  
- too large → loss can explode / diverge  
- good value → loss decreases fast and stabilizes  

What i want is:

- fast decrease of $$J$$  
- stable convergence to a low value  

---

# Note 4 — Hyperparameter tuning is trial + evaluation
In real projects, i usually cannot guess the best hyperparameters immediately.

So the actual workflow looks like:

1) pick a hyperparameter setting  
2) train the model  
3) check results (especially on validation data)  
4) adjust and repeat  

This is an empirical loop:
> try → measure → change → repeat

---

# Note 5 — Typical tuning moves i make
Common adjustments:

- change $$\alpha$$ (0.01 → 0.05, or smaller if unstable)
- increase/decrease number of layers $$L$$
- increase/decrease hidden units $$n^{[l]}$$
- switch activation (ReLU is a common default)

I usually change **one major thing at a time**, so i know what caused the difference.

---

# Note 6 — Using validation set to choose hyperparameters
Hyperparameters should be chosen based on **validation performance**, not training performance.

A typical split:

- training set → learn $$W,b$$  
- validation set → choose hyperparameters  
- test set → final evaluation  

So i compare hyperparameter settings by:

- validation loss  
- validation accuracy (or other metrics)

---

# Note 7 — Quick Python: trying multiple learning rates
```python
import numpy as np

alphas = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
val_loss = {}

for a in alphas:
    # pretend this returns a validation loss after training
    # loss = train_and_eval(alpha=a)
    loss = np.random.random()  # placeholder
    val_loss[a] = loss

best_alpha = min(val_loss, key=val_loss.get)
best_alpha, val_loss[best_alpha]
```

The idea is simple:
- test a small set of values
- pick the one that performs best on validation

---

# Note 8 — Hyperparameters may change over time
Even for the same task, the best hyperparameters can change because:

- data distribution changes  
- model structure changes  
- computing environment changes (hardware, batch size limits, etc.)

So it makes sense to re-check hyperparameters sometimes instead of trusting one “forever setting”.

---
