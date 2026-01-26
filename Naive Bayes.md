# Note 1 Naive Bayes for Tweet Sentiment 

### Step 1) Build counts カウント作る
Split tweets into **positive corpus** and **negative corpus**  
Build **vocabulary V** (all unique words)  
 Count occurrences:  
`count_pos(word)` = times word appears in positive tweets （その単語がポジティブツイートに出る回数）  
`count_neg(word)` = times word appears in negative tweets （その単語がネガティブツイートに出る回数）  

Also count total words:
`N_pos` = total word tokens in positive corpus （ポジティブ側の総単語数 / トークン数）  
`N_neg` = total word tokens in negative corpus  (ネガティブ側の総単語数 / トークン数）  
<img width="914" height="458" alt="image" src="https://github.com/user-attachments/assets/24b8e5e7-c24c-4794-bf60-76004492b7f7" />

Tweet:
> "I’m happy today, I’m learning."

### Idea
Multiply each word’s **positive evidence**:

$$
score=\prod_{w \in tweet}\frac{P(w\mid pos)}{P(w\mid neg)}
$$

### Step-by-step (from the table)
- **I**: 0.20 / 0.20 = **1**
- **am**: 0.20 / 0.20 = **1**
- **happy**: 0.14 / 0.10 = **1.4**
- **today**: not in vocab → **skip**
- **I**: **1**
- **I'm**: **1**
- **learning**: 0.10 / 0.10 = **1**

### Final score
$$
score = 1 \times 1 \times 1.4 \times 1 \times 1 \times 1 = 1.4
$$

### Decision
- **score > 1 → positive**
- **score < 1 → negative**

So **1.4 > 1**, the tweet is **Positive** ✅

```py
import numpy as np

P_pos = {"i": 0.20, "am": 0.20, "happy": 0.14, "learning": 0.10}
P_neg = {"i": 0.20, "am": 0.20, "happy": 0.10, "learning": 0.10}
tweet = ["i", "am", "happy", "today", "i", "am", "learning"]

score = 1.0
for w in tweet:
    if w in P_pos and w in P_neg:     
        score *= (P_pos[w] / P_neg[w])

print("score =", score)
print("Positive" if score > 1 else "Negative")
```
Result
```
score = 1.4000000000000001
Positive
```
---

### 2) Practical version (use logs to avoid overflow)

$$
\log(score)=\sum_{w\in tweet}\Big(\log P(w\mid pos)-\log P(w\mid neg)\Big)
$$

Decision rule:
- `log(score) > 0` → **positive**
- `log(score) < 0` → **negative**

```py
import numpy as np

P_pos = {"i": 0.20, "am": 0.20, "happy": 0.14, "learning": 0.10}
P_neg = {"i": 0.20, "am": 0.20, "happy": 0.10, "learning": 0.10}
tweet = ["i", "am", "happy", "today", "i", "am", "learning"]

log_score = 0.0
for w in tweet:
    if w in P_pos and w in P_neg: 
        log_score += np.log(P_pos[w]) - np.log(P_neg[w])

print("log(score) =", log_score)
print("Positive" if log_score > 0 else "Negative")
```
Result
```
log(score) = 0.33647223662121273
Positive
```


---

# Note 2 Laplacian smoothing

**add 1 to every word first** (pretend it appears at least once) so the probability **won’t become 0**. 
ラプラス平滑化（Laplacian / Laplace smoothing）は、ピエール＝シモン・ラプラスに由来する名前です（「加算1」の考え方＝ラプラスの継起の法則に関連）。 
平滑化とは、出現回数が0の単語でも小さな確率を与えてゼロ確率をなくすことを指し、文章全体の確率が0になってしまうのを防ぎます。 

If a word never appears in a class (count = 0), then its probability becomes **0**.  
When scoring a tweet, multiplying by 0 makes the whole score **collapse to 0**.

Without smoothing:

$$
P(w\mid pos)=\frac{count_{pos}(w)}{N_{pos}},\quad
P(w\mid neg)=\frac{count_{neg}(w)}{N_{neg}}
$$

With **Laplace smoothing** (avoid zero probability):

$$
P(w\mid pos)=\frac{count_{pos}(w)+1}{N_{pos}+|V|},\quad
P(w\mid neg)=\frac{count_{neg}(w)+1}{N_{neg}+|V|}
$$

### Example
Given:

$$ - \(N_{pos}=13\), \(N_{neg}=12\), \(V=8\) $$ 

For **I** (count = 3):

$$
P(I\mid pos)=\frac{3+1}{13+8}=\frac{4}{21}\approx 0.19
$$

$$
P(I\mid neg)=\frac{3+1}{12+8}=\frac{4}{20}=0.20
$$

For **because** in negative (count = 0):

$$
P(because\mid neg)=\frac{0+1}{12+8}=\frac{1}{20}=0.05
$$

```py
A, B = 13, 8
count = 0

print(count / A)
print((count + 1) / (A + B))

```
Result
```
0.0
0.047619047619047616
```

---

### key point

$$
P(w \mid class)=\frac{count(w,class)+1}{N_{class}+V}
$$

- **+1**: no word gets probability 0  
- **+V**: because we added 1 to **all V words**

---

# Note 3 Log Likelihood

### Why use log?
Naive Bayes multiplies lots of small probabilities (like 0.14, 0.10, 0.05).  
Multiplying many small numbers can become **so tiny that the computer rounds it to 0** (underflow).  
So we use **log** to make the math stable.

In probability, the symbol `|` means:
**“given …” / “under the condition that …”**（〜という条件で）

So:

$$
P(A \mid B)
$$

> **The probability that A happens, given that B is true.**

$$
P(\text{happy} \mid \text{pos})
$$

> **Among positive tweets, how often does the word "happy" appear**

$$
P(\text{happy} \mid \text{neg})
$$

> **Among negative tweets, how often does the word "happy" appear


```math
\text{ratio}(\text{happy})
=
\frac{P(\text{happy} \mid \text{pos})}{P(\text{happy} \mid \text{neg})}
```


> **How many times more likely "happy" appears in positive tweets than in negative tweets.**


---


### 1) Word “ratio” (is this word more positive or negative?)
For each word:

$$
ratio(w)=\frac{P(w\mid pos)}{P(w\mid neg)}
$$

np.log(p_pos / p_neg) computes the log ratio（対数比)

- `ratio = 1` → log score is 0 → neutral（ニュートラル） 
- `ratio > 1` → log score is positive → positive evidence（ポジの証拠） 
- `ratio < 1` → log score is negative → negative evidence（ネガの証拠） 

Example:
- happy: 0.14 / 0.10 = 1.4 → positive-leaning

`p_pos = 0.14` means: the conditional probability（条件付き確率）that the word “happy” appears in a positive class（ポジティブクラス） tweet  
`p_neg = 0.10` means: the conditional probability（条件付き確率） that “happy” appears in a negative class（ネガティブクラス） tweet  
`p_pos / p_neg = 0.14 / 0.10 = 1.4` happy” is 1.4 times more likely to appear in positive tweets than in negative tweets, so it is a positive word（ポジ寄りの単語）  

```py
import numpy as np

p_pos = 0.14
p_neg = 0.10

print(p_pos / p_neg)
print(np.log(p_pos / p_neg))
```

Result

```
1.4
0.3364722366212129
```

---


### 2) Log trick: turn “multiply” into “add”

When you multiply many ratios (multiplication / 掛け算),  
the number can become extremely small and collapse to 0 on a computer (underflow / アンダーフロー).  
So we take **log (対数)** and switch to addition (addition / 足し算).

```math
\log(a\cdot b)=\log(a)+\log(b)
```

Instead of multiplying ratios, we **add** their log values.

Define a word evidence score (lambda / ラムダ):

```math
\lambda(w)=\log\left(\frac{P(w\mid pos)}{P(w\mid neg)}\right)
```

to read lambda:

* `λ = 0` → neutral (neutral / 中立)
* `λ > 0` → positive-leaning (positive / ポジ寄り)
* `λ < 0` → negative-leaning (negative / ネガ寄り)

---

### 3) New scoring

Add up the lambdas for the words in the tweet:

```math
\log(\text{score})=\sum_{w\in tweet}\lambda(w)
```

Decision:

* `log(score) > 0` → positive
* `log(score) < 0` → negative

---

```python
import numpy as np

P_pos = {"happy": 0.14, "sad": 0.10}
P_neg = {"happy": 0.10, "sad": 0.15}
tweet = ["happy", "sad"]

log_score = sum(np.log(P_pos[w] / P_neg[w]) for w in tweet)
print(log_score)
print("Positive" if log_score > 0 else "Negative")
```

Result
```
-0.06899287148695127
Negative
```
<img width="953" height="371" alt="image" src="https://github.com/user-attachments/assets/0cb73c2d-916d-4ee7-9cce-9bd5b3fc5999" />

| Concept | What it is | Japanese term | Why use it |
|---|---|---|---|
| **Laplace smoothing** | A fix for **word probabilities** so they never become **0** (add +1 to counts). | ラプラス平滑化 | Prevents **zero probability** when a word never appeared in a class. | 
| **Ratio** | A word’s **evidence multiplier**: how much more the word supports **pos** vs **neg**. | 比率（ひりつ） | Tells if a word is **positive / negative / neutral**. |
| **Lambda (λ)** | The **log version of ratio** (a word’s log evidence score). | ラムダ | Turns multiplication into addition; sign tells sentiment direction. | 
| **Log-likelihood** | The **tweet-level score** computed by **adding lambdas** (instead of multiplying ratios). | 対数尤度（たいすうゆうど） | Avoids underflow; safer for long tweets. |

- Laplace smoothing → makes `P(w|class)` safe (not 0)  
- ratio → compares `P(w|pos)` vs `P(w|neg)`  
- lambda → `log(ratio)`  
- log-likelihood → sum of lambdas for the tweet


---
### 3) Tiny Intuition Example
Assume the tweet has only 3 words:

- **happy**: ratio = **1.4** (positive-leaning)  
- **I**: ratio = **1** (neutral)  
- **sad**: ratio = **0.6** (negative-leaning)

---

#### 1) Original way: multiply ratios
$$
score = 1.4 \times 1 \times 0.6 = 0.84
$$

Since **0.84 < 1**, the tweet is **Negative**.

---

#### 2) Log way: add log ratios (safer)
$$
\log(score)=\log(1.4)+\log(1)+\log(0.6)
$$

Neutral words disappear because:

$$
\log(1)=0
$$

So only the “positive vs negative” words really matter.  
The final sum becomes **negative**, so the tweet is **Negative** 


 ```python 
np.log(x) = ln(x)（自然對數）
np.log10(x) = log10(x)
```
# Note 4 Log Likelihood 2(Training Pipeline)

### What to do
To classify one tweet, just **add up the lambda scores of its words**:

$$
\text{log-likelihood(tweet)}=\sum_{w\in tweet}\lambda(w)
$$

<img width="1001" height="401" alt="image" src="https://github.com/user-attachments/assets/7391259a-26e0-47bd-882d-cf64e73efa9b" />

### Example tweet
**"I am happy because I am learning"**

Given a lambda dictionary:
- λ(I) = 0  
- λ(am) = 0  
- λ(happy) = 2.2  
- λ(because) = 0  
- λ(learning) = 1.1  

--

### Add them up

$$
0+0+2.2+0+0+0+1.1=3.3
$$


```py
tweet = ["I", "am", "happy", "because", "I", "am", "learning"]

lam = {
    "I": 0.0,
    "am": 0.0,
    "happy": 2.2,
    "because": 0.0,
    "learning": 1.1
}

log_score = sum(lam[w] for w in tweet)
print(log_score)          
print(log_score > 0)      
```
Result
```
3.3000000000000003
True
```

---

### Decision rule
Because $\(\log(1)=0\)$, the threshold is **0**:
- **score > 0 → Positive**
- **score < 0 → Negative**

So **3.3 > 0**, the tweet is **Positive** 

---
```py
### Step 1) Collect and label tweets
Split the dataset into two groups:
- **Positive tweets**
- **Negative tweets**

pos_tweets = [
    "I am happy because I am learning",
    "I love NLP"
]

neg_tweets = [
    "I am sad",
    "I do not like this"
]
```
---

### Step 2) Preprocess the text (very important)
Typical cleaning steps:
1. **Lowercase** everything  
2. Remove **punctuation / URLs / @handles**  
3. Remove **stop words** (e.g., *the, is, at*)  
4. Apply **stemming** (e.g., *learning → learn*)  
5. **Tokenize** into words (e.g., `"I am happy" → ["I","am","happy"]`)

> In practice, data cleaning often takes the most time.

```py
import re, string

stopwords = {"i", "am", "because", "do", "not", "this", "the", "is", "at"}

def simple_stem(w):
    for suf in ("ing", "ed", "s"):
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    return w

def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [simple_stem(w) for w in tokens if w not in stopwords]
    return tokens

print(preprocess("I am happy because I am learning"))
]
```

---

### Step 3) Build vocabulary + frequency counts
From the cleaned tweets:
- Build **vocabulary V** (unique words)
- Count word occurrences:
  - `freq(word, pos)`
  - `freq(word, neg)`
Also compute total word counts:
- `N_pos` = total words in positive corpus  
- `N_neg` = total words in negative corpus  

```py
from collections import Counter

freq = Counter()

for t in pos_tweets:
    for w in preprocess(t):
        freq[(w, "pos")] += 1

for t in neg_tweets:
    for w in preprocess(t):
        freq[(w, "neg")] += 1

vocab = sorted({w for (w, _) in freq.keys()})
V = len(vocab)

N_pos = sum(freq[(w, "pos")] for w in vocab)
N_neg = sum(freq[(w, "neg")] for w in vocab)

print("vocab =", vocab)
print("V =", V, "N_pos =", N_pos, "N_neg =", N_neg)

```
---

### Step 4) Compute word probabilities (Laplace smoothing)
Use add-one smoothing to avoid zero probability:

$$
P(w\mid class)=\frac{freq(w,class)+1}{N_{class}+V}
$$

- \(V\) = number of unique words in the vocabulary  
- Ensures every word has **P > 0**
```py
def P(w, label):
    N_class = N_pos if label == "pos" else N_neg
    return (freq[(w, label)] + 1) / (N_class + V)

for w in vocab:
    print(w, "P(w|pos)=", round(P(w, "pos"), 3), "P(w|neg)=", round(P(w, "neg"), 3))
```
<img width="961" height="399" alt="image" src="https://github.com/user-attachments/assets/001436f7-fd09-4b5f-86af-1212457eb399" />

---

### Step 5) Compute each word’s lambda (word sentiment strength)
$$
\lambda(w)=\log\left(\frac{P(w\mid pos)}{P(w\mid neg)}\right)
$$

Interpretation:
- **λ > 0** → positive-leaning word  
- **λ < 0** → negative-leaning word  
- **λ = 0** → neutral word  

```py
import numpy as np

lam = {}
for w in vocab:
    lam[w] = np.log(P(w, "pos") / P(w, "neg"))
    print(w, "lambda =", round(lam[w], 3))

```
---

### Step 6) Compute the log prior (class imbalance term)
Based on tweet counts:

$$
\text{logprior}=\log\left(\frac{N_{pos}}{N_{neg}}\right)
$$

- If classes are balanced → logprior = 0  
- If imbalanced → logprior shifts predictions

```py
logprior = np.log(len(pos_tweets) / len(neg_tweets))
print("logprior =", logprior)

```
---

## NOTE5 Addtional 

| Approach | Typical pipeline | Preprocessing style | Strengths | Weaknesses | Best use case |
|---|---|---|---|---|---|
| **A) Traditional ML** (Naive Bayes / Logistic / SVM) | Clean text → n-grams → Count/TF-IDF → classifier | Usually **some cleaning** (lowercase, URL/handle removal). Often **keep negations** (`not/no/never`). Stemming optional. | Fast, simple, strong baseline, works well on small data, very interpretable | Needs feature engineering, may miss context/sarcasm, limited understanding of word order | Quick baseline, limited compute, classic sentiment tasks |
| **B) Transformers** (BERT/RoBERTa etc.) | Raw text → tokenizer → fine-tune / inference | **Minimal cleaning**. Usually **no stopword removal**, **no stemming**. Keep punctuation/emojis. | Best accuracy, captures context, handles modern language better | Heavier compute, less interpretable, needs more data/tuning | Real-world production, highest performance, complex language |

