<img width="977" height="331" alt="image" src="https://github.com/user-attachments/assets/74fd71ca-1eaf-44af-b79a-302cb6a478b5" /># Note 1 - Co-occurrence Matrix

## What this part is doing
Goal: **turn text into vectors** → so I can compute **similarity** between words / documents.

Two common setups:
1. **word × word**
2. **word × document**

---

## 1) Co-occurrence = “appearing near each other”
Two words **co-occur** if they show up close to each other inside a window.

- **window size = k**
- if `k = 2`, then words within **1–2 tokens apart** count as co-occurring

---

## 2) Word × Word co-occurrence matrix (classic)
I build a matrix **M**:

- row = **target word**
- column = **context word**
- cell value = **how many times context appears near target (within k)**

### Formula

$$
\[
M_{i,j} = \text{count}(w_i \text{ occurs near } w_j \text{ within window } k)
\]
$$

### What this means in my head
A word is basically defined by its **neighbors**.  
More shared neighbors → more similar meaning/usage.

### Python (build word×word co-occurrence)
```python
import numpy as np

text = "I like raw data and simple data"
tokens = text.lower().split()

def build_cooc(tokens, k=2):
    vocab = sorted(set(tokens))
    word2id = {w:i for i,w in enumerate(vocab)}
    n = len(vocab)
    M = np.zeros((n, n), dtype=int)

    for i, w in enumerate(tokens):
        wi = word2id[w]
        left = max(0, i-k)
        right = min(len(tokens), i+k+1)

        for j in range(left, right):
            if j == i:
                continue
            cj = word2id[tokens[j]]
            M[wi, cj] += 1

    return M, vocab, word2id

M, vocab, word2id = build_cooc(tokens, k=2)
vocab, M
```
#　NOTE 2 - Example: vector for data
Assume my context vocab is:

- `simple`
- `raw`
- `like`
- `I`

With window size **k = 2**, I observe:

- `data` near `simple` → **2**
- `data` near `raw` → **1**
- `data` near `like` → **1**
- `data` near `I` → **0**

<img width="977" height="331" alt="image" src="https://github.com/user-attachments/assets/7fc8b167-a9e1-4347-91d1-337319d35c8d" />

So:

$$
\[
\vec{data} = [2,\;1,\;1,\;0]
\]
$$

**Interpretation:**  
`data` = a summary of who it keeps showing up with.

### Python (inspect the vector of `"data"`)
```python
data_vec = M[word2id["data"]]
list(zip(vocab, data_vec))
```

---

## Dimension size
If my vocabulary size is **n**, then each word vector is **n-dimensional**.  
That’s why later we often need **dimensionality reduction**.

---

## 3) Why vectors let me compare similarity
Once everything is a vector, similarity becomes geometry:

- vectors close → distribution is similar → meaning/topic is similar

Two standard metrics:

### (1) Cosine similarity (angle-based)

$$
\[
\cos(\theta)=\frac{\vec a \cdot \vec b}{\|\vec a\|\;\|\vec b\|}
\]
$$


### Python (cosine similarity)
```python
def cosine(a, b, eps=1e-12):
    a = a.astype(float)
    b = b.astype(float)
    return float(a @ b / ((np.linalg.norm(a)+eps) * (np.linalg.norm(b)+eps)))

cosine(M[word2id["data"]], M[word2id["simple"]])
```

### (2) Euclidean distance (distance-based)

$$
\[
d(\vec a,\vec b)=\sqrt{\sum_i (a_i-b_i)^2}
\]
$$

### Python (euclidean distance)
```python
def euclidean(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return float(np.linalg.norm(a - b))

euclidean(M[word2id["data"]], M[word2id["simple"]])
```
<img width="1007" height="477" alt="image" src="https://github.com/user-attachments/assets/343bd778-be82-4282-9505-faff96c1126a" />

---

## Bonus: nearest neighbors (who is most similar to a word?)
### Python (top-k neighbors)
```python
def nearest_neighbors(word, M, vocab, word2id, topk=5):
    idx = word2id[word]
    target = M[idx]
    scores = []
    for w in vocab:
        if w == word:
            continue
        s = cosine(target, M[word2id[w]])
        scores.append((w, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:topk]

nearest_neighbors("data", M, vocab, word2id, topk=5)
```

---

## 4) Word × Document matrix (word-to-category/topic)
This version doesn’t focus on neighbors.  
It asks:

> how often does a word appear in different document categories?

Example categories:
- entertainment
- economy
- machine learning

Example matrix:

| word | entertainment | economy | ML |
|---|---:|---:|---:|
| data | 500 | 6620 | 9320 |
| film | 7000 | 4000 | 1000 |

Two ways to treat vectors:

### (A) Word vectors (take a row)

$$
\[
\vec{data} = [500,\;6620,\;9320]
\]
\[
\vec{film} = [7000,\;4000,\;1000]
\]
$$

### (B) Topic/document vectors (take a column)
Using `data` and `film` as 2D axes:

$$
\[
\vec{entertainment} = [500,\;7000]
\]
\[
\vec{economy} = [6620,\;4000]
\]
\[
\vec{ML} = [9320,\;1000]
\]
$$

### Python (word×document with pandas)
```python
import pandas as pd

df = pd.DataFrame(
    {
        "entertainment": {"data": 500, "film": 7000},
        "economy": {"data": 6620, "film": 4000},
        "ML": {"data": 9320, "film": 1000},
    }
).fillna(0)

df
```

### Python (compare topic vectors)
```python
ent_vec = df["entertainment"].to_numpy()
eco_vec = df["economy"].to_numpy()
ml_vec  = df["ML"].to_numpy()

cosine(eco_vec, ml_vec), cosine(ent_vec, ml_vec)
```

---

# Note 3 Extra (my nlp work)

## My “I got stuck and figured it out” notes (co-occurrence matrix edition)

### 1) My counts looked wrong → it was the window boundary + self-counting
At first my matrix numbers didn’t match my intuition, and it was usually because:

- I accidentally **counted the target word itself** as context  
- I forgot that words near the **beginning/end** of a sentence have **smaller windows**

So my loop must do:

- context range = `[i-k, i+k]` (clipped to valid indices)
- skip `j == i`

```python
left = max(0, i-k)
right = min(len(tokens), i+k+1)

for j in range(left, right):
    if j == i:
        continue
    M[target_id, context_id] += 1
```

---

### 2) “Should the matrix be symmetric?” → depends on how I count
I kept wondering why `M[i,j] != M[j,i]`.

That’s normal if I do **directional counting** (target→context).  
If I want symmetry, I can manually add both directions:

```python
M[wi, cj] += 1
M[cj, wi] += 1   # optional: force symmetry
```

I just need to stay consistent, because **cosine similarity depends on the vector definition**.

---

### 3) My vectors were dominated by boring frequent words → I needed filtering
When I used raw counts, the “top neighbors” became stuff like:

- `the`, `and`, `to`, `is` (or super common tokens)

So I either:

- remove stopwords  
- set `min_count` (drop rare/too common words)
- or upgrade weighting later (TF-IDF / PPMI)

I remind myself: **counts ≠ meaning**, counts are just the starting point.

---

### 4) My `data` vector didn’t look like `[2,1,1,0]` → vocabulary order issue
Even if the counts are correct, the vector can look “wrong” if my vocab ordering is different.

To debug, I always print `(word, value)` pairs:

```python
data_vec = M[word2id["data"]]
list(zip(vocab, data_vec))
```

That way I don’t lie to myself with raw arrays.

---

### 5) “Cosine vs dot product” finally clicked when I saw normalization
I kept mixing these two:

- dot product: \(\vec a \cdot \vec b\)
- cosine similarity: \(\frac{\vec a\cdot\vec b}{\|\vec a\|\|\vec b\|}\)

The shortcut rule I wrote down:

✅ **If vectors are L2-normalized, dot product ≈ cosine similarity**

So if I normalize my vectors first:

```python
def l2norm(v, eps=1e-12):
    return v / (np.linalg.norm(v) + eps)

a_hat = l2norm(a)
b_hat = l2norm(b)
score = a_hat @ b_hat  # this is cosine
```

This is also why in retrieval systems, people normalize embeddings and then just do `emb.dot(query)`.

---

### 6) I hit a division-by-zero bug → some vectors can be all zeros
Sometimes a word gets a zero vector (no valid contexts, filtered out, etc.).  
So I always keep an epsilon:

\[
\hat{v}=\frac{v}{\|v\|+\epsilon}
\]

```python
eps = 1e-12
v = v / (np.linalg.norm(v) + eps)
```

---

### 7) Word×Word vs Word×Document: I confused myself until I wrote this
Both are “co-occurrence”, but the meaning of axes changes.

**Word × Word**
- vector answers: “who does this word appear near?”

**Word × Document**
- vector answers: “which topic/category does this word belong to?”

So when I compare two vectors, I must be clear what I’m comparing:

- word similarity (usage neighbors)
- topic similarity (distribution across categories)

Otherwise I’ll interpret the score totally wrong.

---

### 8) My code was slow when vocab got big → dense matrix was the problem
A full `n × n` matrix explodes fast.

If vocab is large, I should switch to:

- sparse matrices (scipy)
- or store counts in a dict first

But for learning/debugging, I keep a tiny toy example first, verify logic, then scale up.

---

### Quick sanity checklist (before I trust my results)
- [ ] Did I skip `j == i`?
- [ ] Did I clip window boundaries correctly?
- [ ] Is my vocab ordering consistent with my vector interpretation?
- [ ] Am I comparing with cosine, or dot on normalized vectors?
- [ ] Did stopwords dominate the neighbors?
- [ ] Did any vector become zero?

---

