# Note 1 - Co-occurrence

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
# NOTE 2 - Example: vector for data
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
<img width="924" height="349" alt="image" src="https://github.com/user-attachments/assets/77049424-808e-4cb4-9505-cd2261b6703d" />

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

## 0) My mental model (so I don’t get lost)
After the Vector Space Model chapter, I forced myself to summarize my app like this:

- Every JP meme sentence → a vector **doc_vec**
- My English query → a vector **query_vec**
- Similarity score = **cosine similarity**
- Top results = documents with the highest similarity score

So my whole search pipeline is basically:

$$
\[
\text{score}(q,d)=\cos(\vec q,\vec d)
\quad\Rightarrow\quad
\text{pick top-}k
\]
$$

---

## 1) “Why is my dot product called cosine?” → because I normalized everything
At first I thought I **must** implement cosine explicitly:

\[
\cos(\theta)=\frac{\vec q\cdot\vec d}{\|\vec q\|\|\vec d\|}
\]

But I already normalize vectors here:

```python
def norm_rows(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, EPS)
    return x / norms
```

So after normalization, \(\|\vec q\|\approx 1\) and \(\|\vec d\|\approx 1\).  
That means:

✅ **dot product ≈ cosine similarity**

So this line is fine:

```python
scores = emb.dot(query_vec)
```

And my chapter note finally “clicked”:  
**Normalize → dot = cosine → topk retrieval**

---

## 2) My scores looked weird (sometimes too low / too strict) → it was my `min_score`
I added a slider:

```python
min_score = st.slider("Match strictness", 0.0, 1.0, 0.55, 0.01)
```

Then I filter:

```python
cand = [(int(i), float(s)) for i, s in zip(idx, scores) if s >= min_score]
```

What I noticed while testing:
- If I set `min_score` too high, I get **No match** even when results “feel” relevant.
- Especially for short queries / slang, the score is naturally lower.

So my rule became:
- if “No match” happens too often → **lower `min_score` first**
- the model isn’t necessarily wrong, my threshold is.

---

## 3) “MMR gave better variety” → because plain topk repeats the same vibe
When I didn’t use MMR, my results were often **too similar to each other**.

Plain topk:

```python
final_idx = cand_idx[:k].tolist()
```

MMR version:

$$
\[
\text{MMR}=\lambda\cdot \text{relevance}-(1-\lambda)\cdot \text{redundancy}
\]
$$

In code (the core idea):

```python
mmr_scores = lam * rel - (1 - lam) * max_sim
```

What I learned by playing with it:
- **λ bigger** → more accurate but more repetitive
- **λ smaller** → more diverse but sometimes less relevant

So I keep it optional:

```python
use_mmr = st.checkbox("Use MMR...", value=False)
```

---

## 4) “MMR needs more candidates” → it can’t diversify
I almost made the mistake of feeding MMR only the top `k` results.

But MMR needs a **candidate pool** first:

```python
candidate_pool = max(80, k * 10)
idx, scores = search_topk(qv, data["emb"], candidate_pool)
```

My logic:
- MMR is like “picking a good playlist”
- if I only give it 12 songs, it can’t “diversify”
- so I give it a bigger pool, then let it select the final top-k

---

## 5) My app rejected short meme inputs too aggressively →  needed a whitelist
I added a “weird input filter”:

```python
if strict:
    if q_lower not in ALLOW_SHORT and junk(q):
        st.warning("Maybe type a bit longer phrase...")
        st.stop()
```

My original `junk()` rule:

```python
if len(q) < 3:
    return True
```

But meme slang is often **2 letters** (`fr`, `gg`, `idk`, `jk`…), so I created `ALLOW_SHORT`:

```python
ALLOW_SHORT = {"lol","idk","fr","gg","w","l", ...}
```

So now my filter is:
- reject random symbols / empty spam ✅
- still allow real slang ✅

---

## 6) My query embedding shape confused me → I forced it into a 1D vector
When I embed the query, `encode()` returns a batch shape like `(1, dim)`:

```python
v = model.encode([f"query: {q}"], show_progress_bar=False)
v = norm_rows(v)
return v[0]
```

The key part is `v[0]`.

Because later I do:

```python
scores = emb.dot(query_vec)
```

If `query_vec` accidentally stays `(1, dim)`, I risk shape weirdness.
So I always return a clean `(dim,)` vector.

---

## 7) “Why do I need `query:` and `passage:` prefixes?” → E5 expects them
I used:

```python
passages.append(f"passage: {txt}")
v = model.encode([f"query: {q}"])
```

This wasn’t just aesthetic.
My understanding after reading about E5 formatting:
- the model learns “query vs document” roles
- so prefixing helps alignment

When I removed prefixes during testing, results felt less stable.

So I kept the strict format:
- `query: ...`
- `passage: ...`

---

## 8) I got scared of divide-by-zero → EPS is my safety belt
While normalizing, I realized a vector could be all zeros (rare but possible).

So I used:

```python
EPS = 1e-12
norms = np.where(norms > 0, norms, EPS)
```

This is basically:
- “if norm is 0 → replace with EPS”
- avoid crashing
- still return a valid float array

---

## 9) My embeddings took too long to reload → cache saved my life
Computing embeddings is expensive:

```python
emb = model.encode(passages, batch_size=64, show_progress_bar=False)
```

So I cached:
- model once:

```python
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL)
```

- embeddings until CSV changes:

```python
@st.cache_data
def load_data_and_emb(csv_mtime):
    ...
```

And I pass `mtime` so cache invalidates correctly:

```python
mtime = os.path.getmtime(CSV_PATH)
data = load_data_and_emb(mtime)
```

My simple rule:
- if results don’t update after editing CSV → check `mtime` and caching logic first

---

## 10) Quick debugging prints I used (when I was paranoid)
When I felt unsure, I printed only these:

```python
print(data["emb"].shape)      # (N, dim)
print(qv.shape)               # (dim,)
print(scores.min(), scores.max())
print(final_idx[:5])
```

That’s enough to catch 90% of my mistakes without overthinking.

---


