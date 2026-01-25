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

- **add 1 to every word first** (pretend it appears at least once) so the probability **won’t become 0**.

### Why smoothing is needed
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

### Example (from the video)
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

---

### key point

$$
P(w \mid class)=\frac{count(w,class)+1}{N_{class}+V}
$$

- **+1**: no word gets probability 0  
- **+V**: because we added 1 to **all V words**

---

# Note 3 Log Likelihood
### Why do we use log?
Naive Bayes multiplies lots of small probabilities (like 0.14, 0.10, 0.05).  
Multiplying many small numbers can become **so tiny that the computer rounds it to 0** (underflow).  
So we use **log** to make the math stable.

---

### Step 1) Word “ratio” (is this word more positive or negative?)
For each word:

$$
ratio(w)=\frac{P(w\mid pos)}{P(w\mid neg)}
$$

How to read it:
- `ratio = 1` → neutral word (same in both)
- `ratio > 1` → more positive
- `ratio < 1` → more negative

Example:
- happy: 0.14 / 0.10 = 1.4 → positive-leaning

---

### Step 2) Original scoring (multiplication)
A tweet score is the product of all word ratios:

$$
score=\prod_{w\in tweet} ratio(w)
$$

Decision:
- `score > 1` → positive  
- `score < 1` → negative  

---

### Step 3) Log trick: turn “multiply” into “add”
Key rule:

$$
\log(a\cdot b)=\log(a)+\log(b)
$$

So instead of multiplying ratios, we add their log values.

Define a word weight (often called **lambda**):

$$
\lambda(w)=\log\left(\frac{P(w\mid pos)}{P(w\mid neg)}\right)
$$

How to read lambda:
- `λ = 0` → neutral  
- `λ > 0` → positive  
- `λ < 0` → negative  

---

### Step 4) New scoring (addition = safer)
$$
\log(score)=\sum_{w\in tweet}\lambda(w)
$$

Decision:
- `log(score) > 0` → positive  
- `log(score) < 0` → negative  

<img width="953" height="371" alt="image" src="https://github.com/user-attachments/assets/0cb73c2d-916d-4ee7-9cce-9bd5b3fc5999" />

---
### Step 5) Tiny Intuition Example
Assume the tweet has only 3 words:

- **happy**: ratio = **1.4** (positive-leaning)  
- **I**: ratio = **1** (neutral)  
- **sad**: ratio = **0.6** (negative-leaning)

---

### 1) Original way: multiply ratios
$$
score = 1.4 \times 1 \times 0.6 = 0.84
$$

Since **0.84 < 1**, the tweet is **Negative**.

---

### 2) Log way: add log ratios (safer)
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
# Note 4 Log Likelihood 2

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

---

### Decision rule
Because \(\log(1)=0\), the threshold is **0**:
- **score > 0 → Positive**
- **score < 0 → Negative**

So **3.3 > 0**, the tweet is **Positive** 

## 

Note 5 Naive Bayes Training Pipeline (6 Steps)

### Step 1) Collect and label tweets
Split the dataset into two groups:
- **Positive tweets**
- **Negative tweets**

---

### Step 2) Preprocess the text (very important)
Typical cleaning steps:
1. **Lowercase** everything  
2. Remove **punctuation / URLs / @handles**  
3. Remove **stop words** (e.g., *the, is, at*)  
4. Apply **stemming** (e.g., *learning → learn*)  
5. **Tokenize** into words (e.g., `"I am happy" → ["I","am","happy"]`)

> In practice, data cleaning often takes the most time.

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

---

### Step 4) Compute word probabilities (Laplace smoothing)
Use add-one smoothing to avoid zero probability:

$$
P(w\mid class)=\frac{freq(w,class)+1}{N_{class}+V}
$$

- \(V\) = number of unique words in the vocabulary  
- Ensures every word has **P > 0**

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

---

### Step 6) Compute the log prior (class imbalance term)
Based on tweet counts:

$$
\text{logprior}=\log\left(\frac{N_{pos}}{N_{neg}}\right)
$$

- If classes are balanced → logprior = 0  
- If imbalanced → logprior shifts predictions

---

