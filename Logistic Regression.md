
## Note0  Supervised Learning Training Loop

X → model(θ) → Ŷ → compare with Y → cost → update θ → repeat
Turn text into numbers → guess → compare → fix mistakes → finally classify sentiment. 

- たくさんのツイート（X）を用意して、
- 「ポジティブ(1)」か「ネガティブ(0)」かの正解ラベル（Y）も一緒に教える。
- コンピュータはロジスティック（logistic regression）回帰を使って予測し、各ツイートが 1 か 0 か（Ŷ）を推測する。
- コスト関数（cost）」で、予測（Ŷ）が正解（Y）にどれだけ近いかをチェックする。 
- 間違いが多いならパラメータを調整して、もう一度予測して…を繰り返し、cost が最小になるまで学習
- 最終的に新しい文章でも「特徴を取り出す → モデルに入れる → ポジティブ(1)/ネガティブ(0)を判定」できるようになる。
<img width="1014" height="442" alt="image" src="https://github.com/user-attachments/assets/ec0ef82c-ee20-4319-a297-1a9767fe5339" />



```python
import random

X = [
    "I luv donuts",
    "This sucks",
]

Y = [1, 0]

model_skill = 0.3

for training in range(1, 6):
    wrong = 0

    for text, y in zip(X, Y):
        y_hat = y if random.random() < model_skill else 1 - y
        wrong += (y_hat != y)

    cost = wrong
    print(f"Training {training}: cost={cost}, skill={model_skill:.2f}")

    model_skill = min(0.95, model_skill + 0.15)


```
Result
```
Training 1: cost=1, skill=0.30
Training 2: cost=2, skill=0.45
Training 3: cost=0, skill=0.60
Training 4: cost=0, skill=0.75
Training 5: cost=0, skill=0.90
```
---

## Note1 Negative and Positive Frequencies
<img width="2146" height="790" alt="image" src="https://github.com/user-attachments/assets/2a4320bc-4ce0-41dd-aec1-5a960e4c71e5" />

1.Represent a tweet as a **|V|-dimensional 0/1 vector**, marking **1** for words that appear and **0** otherwise.  
2.As **|V| grows**, vectors become **more sparse** (mostly zeros) and the model needs **more parameters**.   
3.More parameters → **longer training time** and **slower prediction**.

```python
tweets = ["I am happy", "I am sad"]

V = sorted(set(" ".join(tweets).lower().split()))
print("Vocab:", V)

def vec(t):
    words = t.lower().split()
    return [1 if w in words else 0 for w in V]

for t in tweets:
    print(t, "->", vec(t))

```
Result
```
Vocab: ['am', 'happy', 'i', 'sad']
I am happy -> [1, 1, 1, 0]
I am sad -> [1, 0, 1, 1]
```
## Note2 Frequency Counts for Sentiment 
Goal: Count how many times each word appears in positive tweets vs negative tweets, then use these counts as features for Logistic Regression.
Steps
  1. Build a vocabulary (V)  *V = all unique words in the corpus (all tweets)
  2. Split tweets into 2 classes Positive (y=1) Negative (y=0)
  3. Count word frequencies per class  
     pos_count(word) = times the word appears in positive tweets  
     neg_count(word) = times the word appears in negative tweets  

| Word     | Positive count (y=1) | Negative count (y=0) |
|----------|---------------------:|---------------------:|
| happy    |                    2 |                    0 |
| am       |                    0 |                    3 |
| learning |                    1 |                    0 |
| bad      |                    0 |                    2 |



## Note3 Frequency dictionary

Before: a tweet was a big vector with size V (very slow).  
Now: a tweet becomes a small vector with size 3 (much faster).  

*A frequency dictionary: It tells you how many times each word appears in positive tweets and negative tweets 

### 1) Build a frequency dictionary `freqs`
Store how often each word appears in each class:

- `PosFreq(1)` = count in **positive tweets**
- `NegFreq(0)` = count in **negative tweets**

Key format:
- `(word, 1) → PosFreq`
- `(word, 0) → NegFreq`

### 2) Example frequency table
| Vocabulary | PosFreq(1) | NegFreq(0) |
|-----------|-----------:|-----------:|
| I         |          3 |          3 |
| am        |          3 |          3 |
| happy     |          2 |          0 |
| because   |          1 |          0 |
| learning  |          1 |          1 |
| NLP       |          1 |          1 |
| sad       |          0 |          2 |
| not       |          0 |          1 |

*A tweet can be represented as **[bias, positive-frequency-sum, negative-frequency-sum]**, enabling Logistic Regression with only **3 features**.
✅ **Key point:** The model can classify a tweet using only **three numbers**, instead of a **huge vector with thousands of dimensions**.

```python
freq = {
    ("i", 1): 3, ("i", 0): 2,
    ("love", 1): 4, ("love", 0): 0,
    ("hate", 1): 0, ("hate", 0): 5,
    ("this", 1): 1, ("this", 0): 4,
}

def tweet_to_3d(tweet):
    words = tweet.lower().split()
    pos_sum = sum(freq.get((w, 1), 0) for w in words)
    neg_sum = sum(freq.get((w, 0), 0) for w in words)
    return [1, pos_sum, neg_sum]  

tweet = "I love this"
print(tweet, "->", tweet_to_3d(tweet))
```
Result
```
I love this -> [1, 8, 6]
```

## Note4 Clean tweets before feeding them into the model

### 1) Stop Words 不要語 / 不要語句
very common words (e.g., *the, is, and*) that usually carry little meaning for NLP tasks.  
They are often removed to reduce noise and speed up processing, especially in **bag-of-words / TF-IDF** models.  
However removing them can hurt tasks like **sentiment** (e.g., *not good*), so it depends on the goal.
Typical handling: use a stop-word list + keep important exceptions like **negations** (*not, never*).  

- **Bag-of-Words (BoW)**: represents text as a vector of word counts (order is ignored).  
- **TF-IDF**: reweights BoW by down-weighting common words and up-weighting informative ones.  

### 2) Stemming 語幹抽出 / 語幹化
reduces words to a rough root form (e.g., *playing → play*, *studies → studi*).  
shrink the vocabulary for **BoW/TF-IDF** models.  
Downsides: stems can look “wrong” and may hurt meaning (not real words).  
Common stemmers: **Porter Stemmer**, **Snowball Stemmer**.  

- **Porter Stemmer**: a classic rule-based English stemmer that aggressively strips suffixes (fast but stems may look unnatural).  
- **Snowball Stemmer**: an improved, more consistent version of Porter, with better rules and support for multiple languages.  
<img width="1930" height="740" alt="image" src="https://github.com/user-attachments/assets/e3429ef4-d275-4e1f-8d89-ba33a2ee3a3d" />


### 3) Punctuation removal 句読点の削除
deletes symbols like `.,!?` to clean text before vectorization.  
*punctuation can sometimes carry meaning in sentiment (e.g., “Great!!!”, “Really?”).  

### 4) Remove handles & URLs
Handles and URLs usually don’t help sentiment → remove them  

### 5) Lowercasing
Treat GREAT / Great / great as the same word  

### 6) Tokenizing (tokenization)  分かち書き
splits text into smaller units called **tokens** (words, subwords, or symbols).  
- Example: `"I love NLP!" → ["I", "love", "NLP", "!"]`  


## Note5 — NLTK  Natural Language Toolkit（自然言語処理ツールキット）
A popular Python library for **NLP learning and prototyping**, 
provides tools for **tokenization, stemming, stopwords, POS tagging, parsing**, and more.  
It also includes **built-in corpora/datasets** (e.g., labeled tweets) for quick experiments.  
Commonly used in courses and baseline pipelines before moving to larger frameworks.  

```python
import re, string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

def preprocess(tweet):
    tweet = re.sub(r'https?://\S+|www\.\S+', '', tweet)
    tok = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True).tokenize(tweet)
    sw = set(stopwords.words('english'))
    stem = PorterStemmer().stem
    return [stem(w) for w in tok if w not in sw and w not in string.punctuation]


print(preprocess("Tuning GREAT AI model!!! @user1 https://example.com"))

```

Result
```
['tune', 'great', 'ai', 'model']
```

## Note6 Logistic Regression Overall 

## 1) Preprocess the tweet (前処理)

Clean the raw tweet into a list of normalized words (**tokens / トークン**): remove **stop words / ストップワード**, **punctuation / 句読点**, **handles / ハンドル**, **URLs / URL**, then apply **stemming / ステミング** and **lowercasing / 小文字化**.

Example result:

```text
["tun", "ai", "great", "model"]
```

```python
tokens = ["tun", "ai", "great", "model"]
```

---

## 2) Extract features (特徴量)

You turn the tokens into a numeric **feature vector / 特徴ベクトル**:

$$
x = [1,\ \text{pos}*{\text{sum}},\ \text{neg}*{\text{sum}}]
$$

* `1` = **bias / バイアス**
* `pos_sum` = total counts from **positive tweets / ポジティブ**
* `neg_sum` = total counts from **negative tweets / ネガティブ**
  (using a **frequency dictionary / 頻度辞書** `freqs[(word,label)]`)

```python
x = [1, 3476, 245]
```

---

## 3) Compute the score (z) (内積)

Logistic regression uses **parameters / パラメータ** (\theta) and computes a **dot product / 内積**:

$$
z = \theta^T x
$$

```python
import numpy as np
theta = np.array([0.00003, 0.00150, -0.00120])
x = np.array([1, 3476, 245])
z = theta @ x
print(theta)
```
Result
```
[ 3.0e-05  1.5e-03 -1.2e-03]
```
---

## 4) Convert score to probability (確率)

Apply the **sigmoid function / シグモイド関数**:

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

This outputs a **probability / 確率** between 0 and 1.

```python
import numpy as np
z = 4.92
p = 1 / (1 + np.exp(-z))
print(p)   
```
Result
```
0.9927537604041685
```
---

## 5) Classify with a threshold (閾値)

Use a **threshold / 閾値** (usually 0.5):

* if $(p \ge 0.5)$ → **positive**
* if $(p < 0.5)$ → **negative**

Equivalent shortcut:

* if $(z \ge 0)$ → **positive**
* if $(z < 0)$ → **negative**

```python
z = 4.92
pred = 1 if z >= 0 else 0
print(pred)  
```
Result
```
1
```
---

## Final picture (行列 X)

Do this for **m tweets / m個のツイート**, stack rows into **matrix X / 行列X** with shape **(m, 3)**.


## Note7  BIAS

### 1) What is BIAS

Bias is the model’s baseline score, which makes it easier to adjust predictions and separate positive vs. negative cases
In Logistic Regression, it’s the intercept (represented by adding a feature `x₀ = 1`).  
It helps the decision boundary move instead of being forced through the origin.  
```
x = [1, pos_sum, neg_sum]
```
```
z = θ0*1 + θ1*pos_sum + θ2*neg_sum
```


## Note8  Sigmoid(Detailed)
Sigmoid squashes any number into 0 to 1  

$$
h(x^{(i)},\theta)=\frac{1}{1+e^{-\theta^T x^{(i)}}}
$$

it basically convert any real-valued score into a probability (0 to 1)

### Symbols
| Symbol | English | Meaning | 中文 | 意義 |
|---|---|---|---|---|
| $h(x^{(i)},\theta)$ | hypothesis / prediction | predicted probability (0 to 1) | 預測機率 | 模型對第 $i$ 筆資料「是正面」的機率 |
| $x^{(i)}$ | feature vector | input numbers describing the example | 特徵向量 | 第 $i$ 則 tweet 的特徵 |
| $\theta$ | parameters / weights | learned importance of each feature | 參數 / 權重 | 模型學到的權重（每個特徵多重要） |
| $\theta^T$ | transpose of $\theta$ | used to compute the dot product | 轉置 | 方便跟 $x^{(i)}$ 做內積 |
| $\theta^T x^{(i)}$ | dot product / score | weighted sum before sigmoid | 內積 / 分數 | 加權總分（可正可負），丟進 sigmoid 變機率 |
| $e$ | Euler’s number | base of natural exponential | 自然常數 | 約 2.718，用在指數運算 |
| $e^{-\theta^T x^{(i)}}$ | exponential term | part of the sigmoid calculation | 指數項 | sigmoid 裡用來「彎曲」機率的部分 |
| $i$ | index | which training example | 索引 | 第幾筆資料 / 第幾則 tweet |

### Quick intuition
- If $\theta^T x^{(i)}$ is big (positive) → $h \approx 1$
- If $\theta^T x^{(i)}$ is very negative → $h \approx 0$
- If $\theta^T x^{(i)} = 0$ → $h = 0.5$

### Sample

1. Original text: `@YMourri and @AndrewYNg are tuning a GREAT AI model`
2. After preprocessing `[tun, ai, great, model]`
3. turn it into 3 numbers 

$$
x^{(i)}=[1,\ 3476,\ 245]
$$

- 1 = bias
- 3476 = sum of positive word frequencies (pos_sum)
- 245 = sum of negative word frequencies (neg_sum)

4. turn it into a probability

$$
\theta=[0.00003,\ 0.00150,\ -0.00120]
$$

5. then computes the score

$$
\theta^{T}x^{(i)}=4.92
$$

## Note7 — Gradient Descent

θ (theta) = the three knobs of the model (the weights for bias, pos_sum, and neg_sum).
J(θ) / Cost / Loss = a score that measures “how bad the model currently is.”

<img width="1039" height="590" alt="image" src="https://github.com/user-attachments/assets/87516bc6-8842-4455-a887-f1a430a51873" />

### Sample

#### 1) Initialize parameters

#### 2) Classify / Predict

$$
h = h(X,\ \theta)
$$

- X = your data matrix (m×3), where each row is a tweet’s [1, pos_sum, neg_sum].
- h = your predicted probabilities (one value between 0 and 1 for each tweet).

#### 3) Measure “how far off am i” → the gradient (Get gradient).

$$
\nabla = \frac{1}{m} X^{T}(h - y)
$$

- y = the ground-truth labels (1 = positive, 0 = negative)
- (h - y) = how much my prediction is off for each tweet


#### 4) then Update theta

$$
\theta = \theta - \alpha \nabla
$$

- α (alpha) = the learning rate: how big a step i take each time when i turn the knobs
- Too large: it may overshoot and diverge.
- Too small: training becomes painfully slow.

#### 5) how bad am i 
then we finally get a

$$
J(\theta)
$$

-  → repeat →  Until good enough


#### 6) Intuition 
 <img width="935" height="392" alt="image" src="https://github.com/user-attachments/assets/1c623c5d-f963-4659-aa79-a5c8e716b95e" />

- Iteration: how many times you repeat the update step.

- Cost (loss): how bad the model currently is.

## Note8 — Validation

#### 1) what is validation set?  
   Data set aside during training is called the validation set

- X_val: the validation set features.
- Y_val: the validation set ground-truth labels.

#### 2) how to do preditction
  Compute probabilities using sigmoid

$$
h = \sigma(X_{\text{val}}\theta)
$$

#### 3) Use threshold 0.5 to convert probability to class label.
for instance: 

- `
0.3 → 0
0.8 → 1
0.5 → 1
`
- eventually ill get  a preditction

$$
\hat{y} = [0, 1, 1, \dots]
$$


#### 4) Get the accuracy
- Compare predictions with true labels.(Y_val)
- Match = 1, mismatch = 0.

$$
\text{correct} = [\hat{y}^{(1)} = y^{(1)},\ \hat{y}^{(2)} = y^{(2)},\ \dots]
$$

- then we got accuracy

$$
\[
\text{accuracy}=\frac{\text{number of correct predictions}}{m}
\]
$$



#### 5) Accuracy measures how often the predictions are correct.  
- accuracy = 0.5 → 50% correct (about as good as random guessing)
- accuracy = 0.8 → 80% correct (pretty good)

## Note8 — Cost Function

## What does each symbol mean in the formula? （式の記号の意味）


$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}\Big[y^{(i)}\log(h^{(i)})+(1-y^{(i)})\log(1-h^{(i)})\Big]
$$


* **$$J(\theta)$$**: total loss / cost function （総損失 / コスト関数）
* **$$m$$**: number of training examples （データ数 / サンプル数）
* **$$y^{(i)}$$**: true label (0 or 1) （正解ラベル）
* **$$h(x^{(i)},\theta)=h^{(i)}$$**: predicted probability of class 1 (often written as $$p$$) （予測確率）
* **$$\log$$**: logarithm （対数）

---

Understand it with numbers (数字で理解)

Assume the true label is $$y=0$$ (negative / ネガティブ):

### Case 1: You also think it’s negative 

$$
p=0.1 \Rightarrow 1-p=0.9
$$

$$
loss=-\log(1-p)=-\log(0.9)\approx 0.105
$$

Small penalty (損失が小さい)

### Case 2: You are very sure it’s positive ❌
$$
p=0.99 \Rightarrow 1-p=0.01
$$

$$
loss=-\log(1-p)=-\log(0.01)\approx 4.605
$$

Huge penalty (損失が爆増)

---

## Numeric Python examples 

```python
import numpy as np

ps = [0.01, 0.1, 0.5, 0.9, 0.99]
print("p    loss(y=1)=-log(p)   loss(y=0)=-log(1-p)")
for p in ps:
    print(f"{p:<4} {(-np.log(p)):<18.4f} {(-np.log(1-p)):<.4f}")
```
Result
```
p    loss(y=1)=-log(p)   loss(y=0)=-log(1-p)
0.01 4.6052             0.0101
0.1  2.3026             0.1054
0.5  0.6931             0.6931
0.9  0.1054             2.3026
0.99 0.0101             4.6052
```
---

## Compute average cost for a batch

```python
import numpy as np
p = np.array([0.3,0.8,0.5,0.2,0.9])
y = np.array([0,  1,  1,  0,   1])
loss = -(y*np.log(p) + (1-y)*np.log(1-p))
print(loss, loss.mean())
```
Result
```
[0.35667494 0.22314355 0.69314718 0.22314355 0.10536052] 0.32029394855698473
```
---

## Note9 — Overview

### 1) Every iteration does this:

- Use the current **θ** to predict → get probabilities **h**
- Measure errors → **(h − y)** (too high / too low shows up here)
- Combine all errors into an update direction → the **gradient**
- Update the parameters:

$$
\theta := \theta - \alpha \cdot \text{gradient}
$$

- **α** = learning rate (step size)
- Repeat until the **cost stops decreasing**.

### 2) General version:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

- Meaning: each parameter **θ** updates a little bit in its own direction.

### 3) Logistic Regression-specific:

$$
\theta_j := \theta_j - \alpha \frac{1}{m}\sum_{i=1}^{m}\left(h^{(i)} - y^{(i)}\right)x_j^{(i)}
$$

- Meaning: Update **θ** by summing **(error × that feature)** over all examples, then averaging it to adjust **θ**.

### 4) vectorized version (much faster):

$$
\theta := \theta - \alpha \frac{1}{m}X^T(h - y)
$$

- Meaning: No for-loop needed — one matrix multiplication does it all.

### 5) Core formulas to remember 

- Predict probabilities with the current parameters:  
  $$h = \sigma(X\theta)$$

- Measure how wrong the predictions are and turn it into an update direction:  
  $$\text{gradient} = \frac{1}{m}X^T(h-y)$$

- Update the parameters by taking a small step to reduce the cost:  
  $$\theta := \theta - \alpha \cdot \text{gradient}$$

Repeat until the cost stops going down.
