# Note 1 - Co-occurrence  (共起 / 共現)

**turn text into vectors** → so I can compute **similarity** between words / documents.

Two common setups:
1. **word × word**
2. **word × document**

```py
sent = "you eat cereal from a bowl".split()
W = 2  

def context(word):
    i = sent.index(word)
    near = []
    for j in range(max(0, i-W), min(len(sent), i+W+1)):
        if j != i:
            near.append(sent[j])
    return near

print("cereal ->", context("cereal"))
print("bowl   ->", context("bowl"))
```
Result
```
cereal -> ['you', 'eat', 'from', 'a']
bowl   -> ['from', 'a']
```
---

## 1) Co-occurrence = “appearing near each other”
Two words **co-occur** if they show up close to each other inside a window.

-If the **window size** is `k`:
- up to `k` words to the left, and
- up to `k` words to the right
of the target word.

```python
sent = "A B C D E".split()
k = 2  # window size

i = sent.index("C")  
neighbors = sent[max(0, i-k):i] + sent[i+1:i+k+1]

print(neighbors)
```

Result

```text
['A', 'B', 'D', 'E']
```

---

## 2) Word × Word co-occurrence matrix (classic)
I build a matrix **M**:

- row = **target word**
- column = **context word**
- cell value = **how many times context appears near target (within k)**

A word is basically defined by its **neighbors**.  
More shared neighbors → more similar meaning/usage.

```python
sents = ["I like simple data".split(), "I prefer simple raw data".split()]
cols  = ["simple","raw","like","I"]
k = 2

counts = {c:0 for c in cols}
for sent in sents:
    if "data" in sent:
        i = sent.index("data")
        neighbors = sent[max(0,i-k):i] + sent[i+1:i+k+1]
        for w in neighbors:
            if w in counts:
                counts[w] += 1

print([counts[c] for c in cols])
```

```text
[2, 1, 1, 0]
```
<img width="977" height="331" alt="image" src="https://github.com/user-attachments/assets/7fc8b167-a9e1-4347-91d1-337319d35c8d" />

## 3) Word-by-Document

- Instead of comparing word vs. word, comparing how often a word appears across different document categories (カテゴリ)    

Entertainment    
Economy    
Machine Learning    

- Then pick a few words and count their occurrences (出現回数) in each category:    

data（データ） appears `500 times` in Entertainment（エンタメ）, `6,620 times` in Economy（経済）, and `9,320 times` in Machine Learning

film（映画） appears `7,000 times` in Entertainment（エンタメ）, `4,000 times` in Economy（経済）, and `1,000 times` in Machine Learning

- Treat each category as a vector:    

Entertainment = `data=500, film=7000`    

Economy = `data=6620, film=4000`    

Machine Learning = `data = 9320, film = 1000`    

```python
docs = {
  "Entertainment": ["data film data", "film film"],
  "Economy": ["data data", "film data"],
  "Machine Learning": ["data data data", "data"]
}
words = ["data","film"]

for cat, texts in docs.items():
    text = " ".join(texts).split()
    print(cat, [text.count(w) for w in words])
```

```text
Entertainment [2, 3]
Economy [3, 1]
Machine Learning [4, 0]
```

---

## 4) Dimension size
If my vocabulary size is **n**, then each word vector is **n-dimensional**.  
That’s why later we often need **dimensionality reduction**.

---

## 5) Euclidean Distance

Used to measure how far apart **two points** or **two vectors**（ベクトル） are. Smaller = more similar; larger = more different.

```math
d(A,B)=\sqrt{\sum_{i=1}^{n}(B_i-A_i)^2}
```

```math
\text{(2D)}\quad d(A,B)=\sqrt{(B_1-A_1)^2+(B_2-A_2)^2}
```

2D = 2 features (条件/特徴)    
nD = n features (条件/特徴)    
Each time you add one more feature, you add one more $((\text{difference})^2)$ term into the distance.    

---

### Example 1 Corpus A vs Corpus B

A = (500, 7000), B = (9320, 1000)

```math
d(A,B)=\sqrt{(9320-500)^2+(1000-7000)^2}
=\sqrt{8820^2+(-6000)^2}
=\sqrt{77792400+36000000}
=\sqrt{113792400}\approx 10667.08
```

<img width="902" height="395" alt="image" src="https://github.com/user-attachments/assets/b40af5c9-77f6-440e-86d0-404dab6ff029" />

```python
import numpy as np

A = np.array([500, 7000])
B = np.array([9320, 1000])

d = np.linalg.norm(B - A)
print(d)

```
Result
```
10667.0833
```
---

### Example 2  boba vs ice-cream

w = (0,4,6), v = (1,6,8)

```math
d(v,w)=\sqrt{(1-0)^2+(6-4)^2+(8-6)^2}
=\sqrt{1^2+2^2+2^2}
=\sqrt{1+4+4}
=\sqrt{9}=3
```
<img width="951" height="290" alt="image" src="https://github.com/user-attachments/assets/1b70da6a-c8e1-4f25-93c2-510e0e75e623" />

```python
import numpy as np

w = np.array([0, 4, 6])
v = np.array([1, 6, 8])

d = np.linalg.norm(v - w)
print(d)
```
Result

```
3.0
```

## 6) Cosine similarity （コサイン類似度）
Once everything is a vector, similarity becomes geometry:
<img width="924" height="349" alt="image" src="https://github.com/user-attachments/assets/77049424-808e-4cb4-9505-cd2261b6703d" />
### Difference 
- **Euclidean distance（ユークリッド距離）** = straight-line distance（距離） between two **vectors（ベクトル）** → sensitive to **magnitude（大きさ）** / document length.    
- **Cosine similarity（コサイン類似度）** = similarity（類似度） based on the **angle（角度）** between two **vectors** → focuses on direction（方向） / ratio, not size.    

---

### Formulas
#### Euclidean distance（ユークリッド距離）
~~~math
d(x,y)=\|x-y\|_2=\sqrt{\sum_i (x_i-y_i)^2}
~~~

#### Cosine similarity（コサイン類似度）
~~~math
\cos(x,y)=\frac{x\cdot y}{\|x\|\|y\|}
~~~

Where:
- $\(x\cdot y\)$ is the **dot product（内積）**
- $\(\|x\|\)$ is the **norm（ノルム）** (often **L2 norm（L2ノルム）**)

---

## Why Euclidean distance can be misleading
If two document vectors differ mainly by length, one can be a scaled version of the other:
~~~math
x = k y \quad (k>0)
~~~
- **Cosine similarity（コサイン類似度）** stays high (same direction → similar distribution).
- **Euclidean distance（ユークリッド距離）** can become large (because magnitude（大きさ） differs).

---

## Example (word-count vectors)

* Food $(Food=(5,15))$
* Agriculture $(Agriculture=(20,40))$
* History $(History=(30,20))$

---

```math
\text{Euclidean distance: } d(u,v)=\sqrt{(u_1-v_1)^2+(u_2-v_2)^2}
```

```math
d(Food,Agriculture)=\sqrt{(5-20)^2+(15-40)^2}
=\sqrt{15^2+25^2}
=\sqrt{850}\approx 29.1548
```

```math
d(Agriculture,History)=\sqrt{(20-30)^2+(40-20)^2}
=\sqrt{(-10)^2+20^2}
=\sqrt{500}\approx 22.3607
```

Euclidean conclusion: (d(Agriculture,History) < d(Food,Agriculture)) → **Agriculture and History look “closer”**.

---

```math
\text{Cosine similarity: } \cos(u,v)=\frac{u\cdot v}{\lVert u\rVert \lVert v\rVert}
```

```math
Food\cdot Agriculture = 5\cdot 20 + 15\cdot 40 = 700
```

```math
\lVert Food\rVert=\sqrt{5^2+15^2}=\sqrt{250},\quad
\lVert Agriculture\rVert=\sqrt{20^2+40^2}=\sqrt{2000}
```

```math
\cos(Food,Agriculture)=\frac{700}{\sqrt{250}\sqrt{2000}}\approx 0.9899
```

```math
Agriculture\cdot History = 20\cdot 30 + 40\cdot 20 = 1400
```

```math
\lVert History\rVert=\sqrt{30^2+20^2}=\sqrt{1300}
```

```math
\cos(Agriculture,History)=\frac{1400}{\sqrt{2000}\sqrt{1300}}\approx 0.8682
```

Cosine conclusion: $( \cos(Food,Agriculture) > \cos(Agriculture,History))$ →     
**Food and Agriculture are “more similar”**     
they point in a more similar direction / proportion

---

```python
import numpy as np

Food = np.array([5, 15])
Agriculture = np.array([20, 40])
History = np.array([30, 20])

def euclid(u, v):
    return np.linalg.norm(u - v)

def cosine(u, v):
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

print("Euclid d(Food,Agriculture) =", euclid(F, A))
print("Euclid d(Agriculture,History) =", euclid(A, H))
print("Cos   cos(Food,Agriculture) =", cosine(F, A))
print("Cos   cos(Agriculture,History) =", cosine(A, H))
```
Result
```
Euclid d(Food,Agriculture) = 29.154759474226502
Euclid d(Agriculture,History) = 22.360679774997898
Cos   cos(Food,Agriculture) = 0.9899494936611665
Cos   cos(Agriculture,History) = 0.8682431421244591
```
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


