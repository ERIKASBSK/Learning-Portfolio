
# Note 1 Supervised Learning Training Loop

X → model(θ) → Ŷ → compare with Y → cost → update θ → repeat
Turn text into numbers → guess → compare → fix mistakes → finally classify sentiment. 

1.Build a vocabulary of all unique words  
2.Convert each tweet into a sparse 0/1 vector showing which words appear   
3.but a large vocabulary makes vectors long and slows training and prediction  


- step 1 たくさんのツイート（X）を用意して、
- 「ポジティブ(1)」か「ネガティブ(0)」かの正解ラベル（Y）も一緒に教える。
- コンピュータはロジスティック（logistic regression）回帰を使って予測し、各ツイートが 1 か 0 か（Ŷ）を推測する。
- コスト関数（cost）」で、予測（Ŷ）が正解（Y）にどれだけ近いかをチェックする。 
- 間違いが多いならパラメータを調整して、もう一度予測して…を繰り返し、cost が最小になるまで学習
- 最終的に新しい文章でも「特徴を取り出す → モデルに入れる → ポジティブ(1)/ネガティブ(0)を判定」できるようになる。
<img width="1014" height="442" alt="image" src="https://github.com/user-attachments/assets/ec0ef82c-ee20-4319-a297-1a9767fe5339" />



```python
import random

x = [
    "I luv donuts",
    "This sucks",
]

# 1=positive, 0=negative

y = [1, 1, 1, 0, 0, 0]
model_skill = 0.3

for traing in range(1,6):
  wrong = 0
  for x, y in zip(X,Y)
    y_hat = y if random.random() < skill else 1 - y
    wrong +=(y_hat != y)

```

---

## HOW IT WORKS 
<img width="2146" height="790" alt="image" src="https://github.com/user-attachments/assets/2a4320bc-4ce0-41dd-aec1-5a960e4c71e5" />

1.Represent a tweet as a **|V|-dimensional 0/1 vector**, marking **1** for words that appear and **0** otherwise.  
2.As **|V| grows**, vectors become **more sparse** (mostly zeros) and the model needs **more parameters**.   
3.More parameters → **longer training time** and **slower prediction**.

```python
import re

tweets = [
    "I am happy because I love NLP",
    "I am sad because rain",
    "NLP is fun"
]

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

vocab = sorted({w for t in tweets for w in tokenize(t)})
word2idx = {w: i for i, w in enumerate(vocab)}

print("Vocabulary size |V| =", len(vocab))
print("Vocab =", vocab)

def vectorize_binary(text, word2idx):
    vec = [0] * len(word2idx)
    words = set(tokenize(text))  
    for w in words:
        if w in word2idx:
            vec[word2idx[w]] = 1
    return vec

x0 = vectorize_binary(tweets[0], word2idx)
print("\nTweet:", tweets[0])
print("Vector:", x0)

ones = sum(x0)
zeros = len(x0) - ones
print(f"Non-zero(1) = {ones}, Zero = {zeros}")

n = len(vocab)
print("\nLogReg params = n + 1 =", n + 1)
```

## Note1 — Frequency Counts for Sentiment 
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

## Note2 — Frequency dictionary

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

## Note3 — Clean tweets before feeding them into the model

### 1) Stop Words 
very common words (e.g., *the, is, and*) that usually carry little meaning for NLP tasks.  
They are often removed to reduce noise and speed up processing, especially in **bag-of-words / TF-IDF** models.  
However removing them can hurt tasks like **sentiment** (e.g., *not good*), so it depends on the goal.
Typical handling: use a stop-word list + keep important exceptions like **negations** (*not, never*).  

- **Bag-of-Words (BoW)**: represents text as a vector of word counts (order is ignored).  
- **TF-IDF**: reweights BoW by down-weighting common words and up-weighting informative ones.  

### 2) Stemming 
reduces words to a rough root form (e.g., *playing → play*, *studies → studi*).  
shrink the vocabulary for **BoW/TF-IDF** models.  
Downsides: stems can look “wrong” and may hurt meaning (not real words).  
Common stemmers: **Porter Stemmer**, **Snowball Stemmer**.  

- **Porter Stemmer**: a classic rule-based English stemmer that aggressively strips suffixes (fast but stems may look unnatural).  
- **Snowball Stemmer**: an improved, more consistent version of Porter, with better rules and support for multiple languages.  


### 3) Punctuation removal
deletes symbols like `.,!?` to clean text before vectorization.  
*punctuation can sometimes carry meaning in sentiment (e.g., “Great!!!”, “Really?”).  

### 4) Remove handles & URLs
Handles and URLs usually don’t help sentiment → remove them  

### 5) Lowercasing
Treat GREAT / Great / great as the same word  

### 6) Tokenizing (tokenization)  
splits text into smaller units called **tokens** (words, subwords, or symbols).  
- Example: `"I love NLP!" → ["I", "love", "NLP", "!"]`  

<img width="1930" height="740" alt="image" src="https://github.com/user-attachments/assets/e3429ef4-d275-4e1f-8d89-ba33a2ee3a3d" />

## Note4 — NLTK  
A popular Python library for **NLP learning and prototyping**, 
provides tools for **tokenization, stemming, stopwords, POS tagging, parsing**, and more.  
It also includes **built-in corpora/datasets** (e.g., labeled tweets) for quick experiments.  
Commonly used in courses and baseline pipelines before moving to larger frameworks.  

```python
import nltk                                # Python library for NLP
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt            # library for visualization
import random                              # pseudo-random number generator
```
## Note5 — Create a feature matrix X for all training tweets
### 1) What is BIAS

Bias is the model’s baseline score, which makes it easier to adjust predictions and separate positive vs. negative cases
In Logistic Regression, it’s the intercept (represented by adding a feature `x₀ = 1`).  
It helps the decision boundary move instead of being forced through the origin.  
- `score = (bias×w0) + (pos_sum×w1) + (neg_sum×w2)`

### 2) The whole process
1. Raw tweet
   `I am Happy Because i am learning NLP @deeplearning`  
2. Preprocessing(remove stopwords / handles / punctuation + stemming + lowercase)
   `[happy, learn, nlp]`
3. Feature Extraction
   `[bias, sum_pos, sum_neg]`

<img width="993" height="1108" alt="image" src="https://github.com/user-attachments/assets/e967a30e-a175-4cc6-9080-aba7a5a19d2c" />

## Note6 — Sigmoid
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

## Note8 — Logistic Regression Cost Function

#### 1) What is Logistic Regression Cost Function
- The **cost function** measures **how wrong** my model is
- It sums up how far the predicted probabilities are from the true labels across all examples, so training can use **gradient descent** to minimize it.

$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}\Big(y^{(i)}\log(h^{(i)}) + (1-y^{(i)})\log(1-h^{(i)})\Big)
$$

- Average loss over **m** examples.
#### 2) Single example intuition
- Single example intuition = understanding how the cost behaves on just one training example.

- Case 1: 

$$ 
\(y=1\)
$$

$$
\[
\text{cost}=-\log(h)
\]
$$

If you predict h = 0.99 → great, you’re almost correct (tiny cost).
If you predict h = 0.01 → you’re confidently wrong (huge cost).

- Case 2:

$$ 
\(y=0\)
$$ 

$$ 
\[
\text{cost}=-\log(1-h)
\]
$$

If you predict h = 0.01 → great, you’re almost correct (tiny cost).
If you predict h = 0.99 → you’re confidently wrong (huge cost).

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
