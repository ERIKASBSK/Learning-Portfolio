
# 1 
```python
import os
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
```

- 1) OS For checking the CSV file exist and to get its last modified time, so Streamlit can reload and recompute embeddings only when the data actually changes.
- 2) numpy For embedding(cosine similarity top-k)
- 3) pandas reading my csv
- 4) streamlit making my webUI without frontend
- 5) sentence_transformers (for numpy)
# 2
```python
MODEL = "intfloat/multilingual-e5-base"
CSV_PATH = "data/jpmemes.csv"
EPS = 1e-12
```
- 1) e5-base
  - Query: `query: <your search text>`
  - Document/Passage: `passage: <your candidate text>`
  - `intfloat/multilingual-e5-small` faster
  - `intfloat/multilingual-e5-base` balanced
  - `intfloat/multilingual-e5-large` better accuracy
  - `intfloat/multilingual-e5-large-instruct`  my pc not allowed me
  I use a pre-trained multilingual embedding model instead of training my own, because full training is computationally expensive and time-consuming.
  This project focuses on retrieval logic and practical semantic matching rather than model training from scratch.
- 2) Eps
  - avoid division by zero when normalizing vectors.
# 3
```python
ALLOW_SHORT = {
    "lol", "lmao", "rofl", "omg", "wtf", "idk", "ikr", "ngl", "fr", "tbh",
    "sus", "cap", "bet", "bruh", "gg", "w", "l", "nsfw", "jk", "pls", "thx", "wtf"
}
``` 
- for the setting/config, could be update at anytime

# 4

```python
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL)
```
- 1) So it loads only once and is reused across Streamlit reruns. I spent ~2 hours debugging slow reruns before realizing caching was the fix....  
  REF : `https://stackoverflow.com/questions/78879594/st-cache-data-is-deprecated-in-streamlit`
        `https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource`
# 5
```python
 def norm_rows(x):  
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, EPS)
    return x / norms       
```     
- `np.asarray(x, dtype=np.float32)` Converts x into a NumPy array and forces float32 to save memory
- `norms = np.linalg.norm(x, axis=1, keepdims=True)` L2 norm, axis=1 = per-row,  keepdims=True( if false it would be like on the same line)
- `norms = np.where(norms > 0, norms, EPS)` replace zero norms with my eps to avoid disivion by zero.

 REF : `https://stackoverflow.com/questions/14415741/what-is-the-difference-between-np-array-and-np-asarray`
# 6
```python
@st.cache_data
def load_data_and_emb(csv_mtime):
    df = pd.read_csv(CSV_PATH)
    if "jp_text" not in df.columns:
        st.error("No 'jp_text' column in CSV...")
        st.stop()

    df["jp_text"] = df["jp_text"].fillna("").astype(str)

    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    else:
        df["source"] = ""

    if "img" in df.columns:
        df["img"] = df["img"].fillna("").astype(str)
    else:
        df["img"] = ""

    passages = []
    for idx, row in df.iterrows():
        txt = row["jp_text"].strip()
        passages.append(f"passage: {txt}")

    model = load_model()
    emb = model.encode(passages, batch_size=64, show_progress_bar=False)
    emb = norm_rows(emb)

    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
        "img": df["img"].to_numpy(),
    } 
 ```
- `@st.cache_data def load_data_and_emb(csv_mtime):` update it and rerun the calculation anytime
- `df = pd.read_csv(CSV_PATH)` panda
- `if "jp_text" not in df.columns:
        st.error("No 'jp_text' column in CSV...")
        st.stop()` avoid crashing
- `    if "source" in df.columns:
        df["source"] = df["source"].fillna("").astype(str)
    else:
        df["source"] = ""`  
  fillna("") avoid NaN, astype(str) to be str, if no source column then create it
- `    if "img" in df.columns:
        df["img"] = df["img"].fillna("").astype(str)
    else:
        df["img"] = ""` same with image but so far i havent fully maintained
- `passages = [] for idx, row in df.iterrows():
    txt = row["jp_text"].strip()
    passages.append(f"passage: {txt}")` for the e5 base, iterate through a DataFrame row by row, remove the blank

- `model = load_model()` read my @st.cache_resource
- `emb = model.encode(passages, batch_size=64, show_progress_bar=False)` convert, 64 sentences per batch,hide the progress bar(for not looking too hideous)
- ` emb = norm_rows(emb)` Normalize embeddings to unit length so dot product equals cosine similarity(normaliaztion)
- `    return {
        "emb": emb,
        "jp_text": df["jp_text"].to_numpy(),
        "source": df["source"].to_numpy(),
        "img": df["img"].to_numpy(),
    } `
  dictionarilized(load_data_and_emb)
# 7
```python
def embed_query(model, q):
    q = q.strip()
    v = model.encode([f"query: {q}"], show_progress_bar=False)
    v = norm_rows(v)
    return v[0]
 ```    
- `q = q.strip() ` remove the blank of input
- `v = model.encode([f"query: {q}"], show_progress_bar=False)` encode the query into an embedding vector
- `v = norm_rows(v)` same as ` emb = norm_rows(emb)`
# 8
```python
def search_topk(query_vec, emb, k):
    scores = emb.dot(query_vec)
    k = min(k, len(scores))
    if k <= 0:
        return np.array([]), np.array([])
    top_idx = np.argsort(scores)[::-1][:k]
    return top_idx, scores[top_idx]
 ```
- `emb.dot(query_vec) ` calculating the score
incase users want more than i have...maybe not that usaful
- `k = min(k, len(scores))
    if k <= 0:
        return np.array([]), np.array([])`
if k=0 then reture nothing
-`top_idx = np.argsort(scores)[::-1][:k]`
argsort() sorts ascending by default, use [::-1] cab reverse it*
# 9
```python
def mmr_select(query_vec, emb, candidate_idx, k, lam=0.75):
    candidate_idx = list(map(int, candidate_idx))
    if not candidate_idx or k <= 0:
        return []

    rel = emb[candidate_idx].dot(query_vec)

    selected = []

    while len(selected) < k and candidate_idx:
        if not selected:
            best_pos = int(np.argmax(rel))
        else:
            cand_emb = emb[candidate_idx]     
            sel_emb = emb[selected]           
  
            sim_to_selected = cand_emb.dot(sel_emb.T)   
            max_sim = sim_to_selected.max(axis=1)       

            mmr_scores = lam * rel - (1 - lam) * max_sim
            best_pos = int(np.argmax(mmr_scores))

        best_idx = candidate_idx[best_pos]
        selected.append(best_idx)

        
        candidate_idx.pop(best_pos)
        rel = np.delete(rel, best_pos)

    return selected
 ```
Goal: keep results relevant to the query, but avoid near-duplicates
I adapted this from online MMR examples and modified it for this project, but it's still under testing and tuning (sowey plz ignore this part)  
Ref:`https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results`  
Ref:`https://developers.llamaindex.ai/python/examples/vector_stores/simpleindexdemommr`
# 10
```python
def junk(q: str) -> bool:
    q = q.strip()
    if len(q) < 3:
        return True

    alnum_count = sum(1 for c in q if c.isalnum())
    threshold = max(2, int(len(q) * 0.3))

    return alnum_count < threshold
```
- `q = q.strip()` remove the blabk
- `if len(q) < 3:
        return True` 
 too short, if ture it is junk
- `    alnum_count = sum(1 for c in q if c.isalnum())`  set the barrrrrr c is the char, sum how many alphanumeric
- `    threshold = max(2, int(len(q) * 0.3))` set the bar-2, maximum 2 alphanumerics or some texts contain more than 2 alphanumerics

##### ========== UI ==================================
# 11
```python
st.set_page_config(page_title="EN → JP Meme Search", layout="wide")
st.title("EN → JP meme search")
```

- page setting   

Ref: `https://docs.streamlit.io/develop/api-reference`
# 12
```python
if not os.path.exists(CSV_PATH):
    st.error(f"CSV file not found: {CSV_PATH}")
    st.stop()
```
simple guard clause

# 13
```python
mtime = os.path.getmtime(CSV_PATH)
data = load_data_and_emb(mtime)
```
Pass CSV mtime as the cache key, so embeddings are reused unless the file gets updated. So I dont have to read csv everytime while typing or using slider, very important feature.  
# 14
```python
with st.sidebar:
    st.header("Settings")
    k = st.slider("How many results", 1, 50, 12)
    min_score = st.slider("Match strictness", 0.0, 1.0, 0.55, 0.01)
    strict = st.checkbox("Ignore weird input", value=True)
    use_mmr = st.checkbox("Use MMR(Under testing. Please refrain from use for now.)", value=False)
    mmr_lam = st.slider("MMR lambda", 0.50, 0.95, 0.75, 0.01)
```
- min_score default value: 12
- `st.header` left side title
- `strict` : refer to #17
- `use_mmr`: refer to #21
- `mmr_lam `: refer to #21
# 15
```python
q = st.text_input(
    "Type English meme/slang",
    placeholder="cringe / no cap / that's so real / touch grass / delulu ..."
)
```
- searching box setting

# 16 
```python
if st.button("Search", type="primary"):
    q = q.strip()

    if not q:
        st.warning("Type something pls")
        st.stop()
```
- `if st.button("Search", type="primary"):` button setting
- if input doesnt exist, show warning
# 17
```python
    q_lower = q.lower()
    if strict:
        if q_lower not in ALLOW_SHORT and junk(q):
            st.warning("Maybe type a bit longer phrase...")
            st.stop()
```
-`  q_lower = q.lower()` Convert the input query to lowercase to avoid case sensitivity affecting the results
- refer to #14 and #3

# 18
```python
model = load_model()
qv = embed_query(model, q)
```
- Load the embedding model (cached) and encode the user query into a normalized vector for similarity search.
- ref to # 7 and #2

# 19

```python
    candidate_pool = max(80, k * 10)
    idx, scores = search_topk(qv, data["emb"], candidate_pool)
```
`candidate_pool = max(80, k * 10)`: Grab a larger candidate set than k so filtering/MMR still has room to choose from
`idx, scores = search_topk(...)`: Compute cosine similarity and return the top-N result indices and their scores

# 20

```python
    cand = [(int(i), float(s)) for i, s in zip(idx, scores) if s >= min_score]

    if not cand:
        st.info("No match. Try other words or lower the strictness.")
        st.stop()

    cand_idx = np.array([i for i, _ in cand], dtype=int)
```
`cand = [...] if s >= min_score`：Pair up (idx, score) and keep only candidates whose score passes the min_score threshold

`if not cand: ... st.stop()`：If nothing survives the filter, show “No match” and stop the search flow

`cand_idx = np.array([...])`：Extract only the remaining indices (drop the scores) and store them as an int numpy array for later MMR

# 21
```py
    if use_mmr:
        final_idx = mmr_select(qv, data["emb"], cand_idx, k=k, lam=mmr_lam)
    else:
        final_idx = cand_idx[:k].tolist()

    if not final_idx:
        st.info("No match after filtering. Try lower strictness.")
        st.stop()
```
`if use_mmr: ...`：If enabled, MMR reranks the candidates to avoid near-duplicate results and returns up to k diverse picks.
`else: `：If MMR is off, just take the first k indices (highest scores) as the final results.
`if not final_idx: ...`：If nothing remains after filtering, show a message and stop the app flow

# 22 

```py
    for rank, i in enumerate(final_idx, 1):
        jp_text = data["jp_text"][i]
        source_tag = data["source"][i]
        img = str(data["img"][i]).strip()

        score = float(data["emb"][i].dot(qv))

        with st.container():
            st.markdown(f"### Rank {rank} — match {score*100:.1f}%")

            if img:
                try:
                    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
                    st.image(img, width=320)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception:
                    st.caption("🖼️ image link broken / blocked")

            st.write(jp_text)

            if source_tag:
                st.caption(f"🏷️ {source_tag}")

            st.divider()
```
- `for rank, i in enumerate(final_idx, 1):` Loop through the final picks and give each one a rank number starting from 1.
- `jp_text / source_tag / img = ...` Grab the JP text, source tag, and image link for that result.
- `score = float(data["emb"][i].dot(qv))` Recompute the cosine match score (dot product),  just for display.
- `with st.container():` Wrap one result into its own nice little UI block
- `st.markdown(...)` Print the rank + match percentage as the result title.
- `if img: try ... except ...` If there’s an image link, show it; if it’s broken, just show a small warning instead of crashing.
- `st.write(jp_text)` Show the actual meme text.
- `if source_tag: st.caption(...)` If there’s a source label, show it under the text.
- `st.divider()` Draw a line to separate this result from the next one.

