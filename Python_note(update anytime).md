# Python Quick  Notebook

---

## Table 
- Basics
- Data Types
- Operators
- Strings
- Lists / Tuples
- Dict / Set
- Comprehensions
- Functions
- Control Flow
- Exceptions
- File I/O
- Modules / Imports
- OOP
- Dataclasses
- Typing
- Iterators / Generators
- Context Managers
- Async / Await
- Standard Library (most used)
- CLI (argparse)
- Logging
- Regex
- JSON / CSV
- Dates & Time
- Pathlib
- Common Patterns
- Debugging
- Testing (pytest)
- Performance Tips
- Useful Snippets
- 

---

# 1) Basics

## 1.1 Comments
```py
# single-line comment

"""
multi-line string (often used as docstring)
"""
````

## 1.2 Print

```py
print("hello")
print("a", 123, True, sep=" | ", end="\n")
```

## 1.3 f-strings (most common)

```py
name = "Erika"
score = 9.5
print(f"{name=} {score:.2f}")
```

## 1.4 Basic input

```py
x = input("Enter: ")       # always str
n = int(input("Number: ")) # parse
```

---

# 2) Data Types (Quick)

## 2.1 Numbers

```py
a = 10        # int
b = 3.14      # float
c = 2 + 3j    # complex
```

### Int division vs float division

```py
7 / 2   # 3.5
7 // 2  # 3
7 % 2   # 1
divmod(7, 2)  # (3, 1)
```

## 2.2 Booleans

```py
True, False
bool(0), bool(""), bool([])   # False False False
bool(1), bool("x"), bool([0]) # True True True
```

## 2.3 None

```py
x = None
x is None      # True
x == None      # avoid (use is)
```

---

# 3) Operators

## 3.1 Arithmetic

```py
+  -  *  /  //  %  ** 
```

## 3.2 Comparison

```py
== != > < >= <=
```

## 3.3 Boolean logic

```py
and or not
```

### Short-circuit

```py
a and b  # if a is falsy, returns a
a or b   # if a is truthy, returns a
```

## 3.4 Identity / Membership

```py
x is y
x is not y
x in container
x not in container
```

## 3.5 Bitwise

```py
& | ^ ~ << >>
```

---

# 4) Strings

## 4.1 Quotes

```py
s1 = "hello"
s2 = 'hello'
s3 = """multi
line"""
```

## 4.2 Common methods

```py
s.lower()
s.upper()
s.title()
s.strip()         # trim whitespace
s.replace("a","b")
s.split(",")
",".join(list_of_str)
s.startswith("he")
s.endswith("lo")
s.find("x")       # -1 if not found
s.index("x")      # raises ValueError if not found
```

## 4.3 Slicing

```py
s = "abcdef"
s[0]     # 'a'
s[-1]    # 'f'
s[1:4]   # 'bcd'
s[:3]    # 'abc'
s[3:]    # 'def'
s[::2]   # 'ace'
s[::-1]  # reverse
```

## 4.4 Formatting (old vs new)

```py
"{} {}".format("a", 1)
f"a {1}"
```

## 4.5 Escape sequences

```py
"\n"  # newline
"\t"  # tab
"\""  # quote
r"\n" # raw string (useful for regex, Windows paths)
```

---

# 5) Lists / Tuples

## 5.1 Create

```py
lst = [1, 2, 3]
tup = (1, 2, 3)
single = (1,)      # IMPORTANT
empty_t = ()
```

## 5.2 Indexing / slicing

```py
lst[0]
lst[-1]
lst[1:3]
```

## 5.3 Common list ops

```py
lst.append(x)
lst.extend([a,b])
lst.insert(i, x)
lst.remove(x)   # remove by value
lst.pop()       # remove last
lst.pop(i)
lst.clear()
```

## 5.4 Sorting

```py
lst.sort()                     # in-place
sorted_lst = sorted(lst)       # new list
sorted(lst, key=len)
sorted(lst, reverse=True)
```

### Sort by multiple keys

```py
sorted(items, key=lambda x: (x["age"], x["name"]))
```

## 5.5 Copying (shallow vs deep)

```py
a = [1,2,[3]]
b = a.copy()         # shallow copy
import copy
c = copy.deepcopy(a) # deep copy
```

---

# 6) Dict / Set

## 6.1 Dict basics

```py
d = {"a": 1, "b": 2}
d["a"]           # 1
d.get("x")       # None
d.get("x", 0)    # default
```

## 6.2 Dict updates

```py
d["c"] = 3
d.update({"a": 10})
d.setdefault("k", 999)
```

## 6.3 Iteration

```py
for k in d:
    ...

for k, v in d.items():
    ...
```

## 6.4 Useful methods

```py
d.keys()
d.values()
d.items()
d.pop("a")
d.popitem()
```

## 6.5 Set basics

```py
s = {1,2,3}
s.add(4)
s.remove(2)    # error if missing
s.discard(2)   # safe
```

### Set ops

```py
a | b  # union
a & b  # intersection
a - b  # difference
a ^ b  # symmetric diff
```

---

# 7) Comprehensions

## 7.1 List comprehension

```py
[x*x for x in range(5)]
[x for x in nums if x % 2 == 0]
```

## 7.2 Dict comprehension

```py
{k: v*v for k, v in d.items()}
```

## 7.3 Set comprehension

```py
{x.lower() for x in words}
```

## 7.4 Generator expression (lazy)

```py
gen = (x*x for x in range(10))
```

---

# 8) Functions

## 8.1 Define

```py
def add(a, b):
    return a + b
```

## 8.2 Default args

```py
def greet(name="world"):
    return f"hi {name}"
```

## 8.3 Keyword-only args

```py
def f(a, *, b, c=0):
    return a + b + c
# f(1, 2) ❌
# f(1, b=2) ✅
```

## 8.4 *args / **kwargs

```py
def f(*args, **kwargs):
    print(args)
    print(kwargs)

f(1,2,3, x=10, y=20)
```

## 8.5 Unpacking

```py
a, b, c = [1,2,3]
x, *rest = [1,2,3,4]
*d1, last = [10,20,30]
```

## 8.6 Return multiple values

```py
def pair():
    return 1, 2

x, y = pair()
```

## 8.7 Lambda (small inline)

```py
key_fn = lambda x: x["score"]
```

## 8.8 Docstrings

```py
def foo():
    """Do something."""
    ...
```

---

# 9) Control Flow

## 9.1 if / elif / else

```py
if x > 0:
    ...
elif x == 0:
    ...
else:
    ...
```

## 9.2 for loop

```py
for i in range(5):
    ...

for i, v in enumerate(lst):
    ...
```

## 9.3 while loop

```py
while cond:
    ...
```

## 9.4 break / continue

```py
for x in xs:
    if x < 0:
        continue
    if x == 0:
        break
```

## 9.5 match/case (Python 3.10+)

```py
match x:
    case 0:
        ...
    case 1 | 2:
        ...
    case str() as s:
        ...
    case _:
        ...
```

---

# 10) Exceptions

## 10.1 try / except

```py
try:
    x = int("abc")
except ValueError as e:
    print(e)
```

## 10.2 finally

```py
try:
    ...
finally:
    cleanup()
```

## 10.3 raise

```py
raise ValueError("bad input")
```

## 10.4 Custom exception

```py
class MyError(Exception):
    pass
```

---

# 11) File I/O

## 11.1 Read text

```py
with open("a.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

## 11.2 Write text

```py
with open("a.txt", "w", encoding="utf-8") as f:
    f.write("hello\n")
```

## 11.3 Read lines

```py
with open("a.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
```

## 11.4 Binary read/write

```py
with open("a.bin", "rb") as f:
    b = f.read()

with open("a.bin", "wb") as f:
    f.write(b"\x00\xFF")
```

---

# 12) Modules / Imports

## 12.1 Basics

```py
import math
from math import sqrt
import numpy as np
```

## 12.2 Aliasing

```py
import pandas as pd
```

## 12.3 Import from file

Project:

```
myproj/
  main.py
  utils.py
```

```py
# main.py
from utils import helper
```

---

# 13) OOP (Classes)

## 13.1 Basic class

```py
class User:
    def __init__(self, name):
        self.name = name

    def hello(self):
        return f"hi {self.name}"
```

## 13.2 Inheritance

```py
class Animal:
    def speak(self):
        return "..."

class Cat(Animal):
    def speak(self):
        return "meow"
```

## 13.3 @property

```py
class A:
    def __init__(self, x):
        self._x = x

    @property
    def x(self):
        return self._x
```

---

# 14) Dataclasses

## 14.1 dataclass

```py
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int
```

## 14.2 Frozen (immutable)

```py
from dataclasses import dataclass

@dataclass(frozen=True)
class Token:
    text: str
    pos: str
```

---

# 15) Typing (Type Hints)

## 15.1 Basics

```py
def add(a: int, b: int) -> int:
    return a + b
```

## 15.2 Common types

```py
from typing import List, Dict, Tuple, Optional

x: List[int] = [1,2,3]
d: Dict[str, int] = {"a": 1}
t: Tuple[int, str] = (1, "x")
o: Optional[int] = None
```

## 15.3 Modern typing (3.9+)

```py
def f(x: list[int]) -> dict[str, int]:
    ...
```

## 15.4 Union (3.10+)

```py
def f(x: int | None) -> str:
    ...
```

---

# 16) Iterators / Generators

## 16.1 Iterator protocol

```py
it = iter([1,2,3])
next(it)
```

## 16.2 Generator

```py
def gen():
    yield 1
    yield 2
```

## 16.3 Generator with filter

```py
(x for x in nums if x > 0)
```

---

# 17) Context Managers

## 17.1 with open(...)

```py
with open("a.txt") as f:
    ...
```

## 17.2 Custom context manager

```py
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    t0 = time.time()
    try:
        yield
    finally:
        print(time.time() - t0)
```

---

# 18) Async / Await (quick)

## 18.1 Define async function

```py
import asyncio

async def main():
    await asyncio.sleep(1)
    return 123

asyncio.run(main())
```

## 18.2 Gather

```py
async def job(i):
    await asyncio.sleep(0.1)
    return i

async def main():
    res = await asyncio.gather(*(job(i) for i in range(10)))
    print(res)
```

---

# 19) Standard Library (Most Used)

## 19.1 math

```py
import math
math.sqrt(9)
math.ceil(3.2)
math.floor(3.8)
math.log(10)
```

## 19.2 statistics

```py
import statistics as st
st.mean([1,2,3])
st.median([1,2,3])
```

## 19.3 random

```py
import random
random.random()
random.randint(1, 10)
random.choice(["a","b","c"])
random.shuffle(lst)
```

## 19.4 collections

```py
from collections import Counter, defaultdict, deque

Counter("banana")                # counts
dd = defaultdict(int)
dd["x"] += 1

q = deque([1,2,3])
q.append(4)
q.popleft()
```

## 19.5 itertools

```py
import itertools as it

list(it.chain([1,2],[3,4]))
list(it.product([1,2], ["a","b"]))
list(it.combinations([1,2,3], 2))
```

## 19.6 functools

```py
from functools import lru_cache, reduce

@lru_cache(maxsize=None)
def fib(n):
    if n < 2: return n
    return fib(n-1) + fib(n-2)

reduce(lambda a,b: a+b, [1,2,3], 0)
```

## 19.7 dataclasses / typing / pathlib (see other sections)

---

# 20) CLI (argparse)

```py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="world")
args = parser.parse_args()
print(f"hi {args.name}")
```

---

# 21) Logging

```py
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logging.info("hello")
logging.warning("warn")
logging.error("error")
```

---

# 22) Regex

```py
import re

s = "abc123xyz"
re.search(r"\d+", s).group()           # "123"
re.findall(r"[a-z]+", s)               # ["abc", "xyz"]
re.sub(r"\d+", "#", s)                 # "abc#xyz"

# groups
m = re.search(r"(\w+)(\d+)", "foo42")
m.group(0) # "foo42"
m.group(1) # "foo"
m.group(2) # "42"
```

Tip: raw string `r"..."` recommended.（正規表現は raw string が安全）

---

# 23) JSON / CSV

## 23.1 JSON

```py
import json

data = {"a": 1, "b": [1,2]}
s = json.dumps(data, ensure_ascii=False)
obj = json.loads(s)
```

## 23.2 CSV

```py
import csv

with open("a.csv", "r", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

with open("b.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["x","y"])
    writer.writeheader()
    writer.writerows([{"x":1,"y":2}])
```

---

# 24) Dates & Time

```py
from datetime import datetime, date, timedelta

now = datetime.now()
today = date.today()
tomorrow = today + timedelta(days=1)

now.strftime("%Y-%m-%d %H:%M:%S")
```

Parse:

```py
dt = datetime.strptime("2026-01-19", "%Y-%m-%d")
```

Timezone (basic):

```py
from datetime import timezone
utc_now = datetime.now(timezone.utc)
```

---

# 25) Pathlib (modern file paths)

```py
from pathlib import Path

p = Path("data") / "file.txt"
p.exists()
p.parent
p.name
p.suffix
p.read_text(encoding="utf-8")
p.write_text("hello", encoding="utf-8")
```

List:

```py
for f in Path(".").glob("*.py"):
    print(f)
```

---

# 26) os / sys (quick)

```py
import os, sys

os.getcwd()
os.environ.get("HOME")
sys.argv
sys.path
```

---

# 27) Common Patterns

## 27.1 Safe dict access

```py
val = d.get("key", default_value)
```

## 27.2 Swap variables

```py
a, b = b, a
```

## 27.3 Count frequencies

```py
from collections import Counter
cnt = Counter(words)
cnt.most_common(10)
```

## 27.4 Unique while preserving order

```py
seen = set()
out = []
for x in items:
    if x not in seen:
        seen.add(x)
        out.append(x)
```

## 27.5 Flatten list of lists

```py
flat = [x for sub in lol for x in sub]
```

## 27.6 Chunk list

```py
def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]
```

## 27.7 Quick timer

```py
import time
t0 = time.time()
# do something
elapsed = time.time() - t0
```

---

# 28) Debugging

## 28.1 print debugging

```py
print("x=", x)
```

## 28.2 breakpoint (Python 3.7+)

```py
breakpoint()
```

## 28.3 pprint

```py
from pprint import pprint
pprint(big_object)
```

---

# 29) Testing (pytest quick)

```py
def add(a, b):
    return a + b

def test_add():
    assert add(1,2) == 3
```

Run:

```bash
pytest -q
```

---

# 30) Performance Tips (practical)

* Prefer **list/dict comprehension** over manual loops (often faster)
* Use **set** for membership tests: `x in set_obj` (fast)
* Avoid repeated string concat in loops; use `''.join(...)`
* For heavy numeric: use **numpy** (vectorized)

---

# 31) Numpy (mini quick)

```py
import numpy as np

a = np.array([1,2,3], dtype=float)
a.mean()
a / (np.linalg.norm(a) + 1e-12)

# matrix
M = np.array([[1,2],[3,4]])
M.T
np.dot(M, M)
```

---

# 32) Pandas (mini quick)

```py
import pandas as pd

df = pd.read_csv("x.csv")
df.head()
df["col"].value_counts()
df.dropna()
df.fillna(0)

df[df["score"] > 0.5]
df.sort_values("score", ascending=False)

df.to_csv("out.csv", index=False)
```

---

# 33) Streamlit (mini quick)

```py
import streamlit as st

st.title("App")
q = st.text_input("Query")
if st.button("Search"):
    st.write(q)
```

Cache:

```py
@st.cache_resource
def load_model():
    ...
```

---

# 34) Common “I Forgot This” Corner

## 34.1 Truthy / Falsy

Falsy:

* `0`, `0.0`
* `""`
* `[]`, `{}`, `set()`
* `None`
* `False`

## 34.2 `==` vs `is`

```py
x == y   # value equality
x is y   # identity (same object)
```

## 34.3 Mutable default argument (DON’T)

```py
def f(x, cache=[]):  # ❌ shared across calls
    cache.append(x)
    return cache
```

Do:

```py
def f(x, cache=None):
    if cache is None:
        cache = []
    cache.append(x)
    return cache
```

## 34.4 Copy list/dict quickly

```py
new_list = old_list[:]      # shallow
new_dict = dict(old_dict)   # shallow
```

## 34.5 `zip` and `*` unzip

```py
a = [1,2]
b = ["x","y"]
list(zip(a,b))   # [(1,'x'), (2,'y')]

pairs = [(1,"x"), (2,"y")]
xs, ys = zip(*pairs)
```

## 34.6 `enumerate`

```py
for i, v in enumerate(lst, start=1):
    ...
```

## 34.7 `any` / `all`

```py
any([False, True])  # True
all([True, True])   # True
```

## 34.8 `map` / `filter`

```py
list(map(int, ["1","2"]))
list(filter(lambda x: x > 0, [-1,0,2]))
```

---

# 35) Useful Snippets (Copy-Paste)

## 35.1 Read a large file line-by-line

```py
with open("big.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
```

## 35.2 Safe parse int

```py
def safe_int(s, default=0):
    try:
        return int(s)
    except ValueError:
        return default
```

## 35.3 Top-K with heapq

```py
import heapq
topk = heapq.nlargest(5, items, key=lambda x: x["score"])
```

## 35.4 Deduplicate with Counter threshold

```py
from collections import Counter
cnt = Counter(words)
keep = [w for w in words if cnt[w] >= 2]
```

## 35.5 Simple cosine similarity

```py
import numpy as np

def cosine(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a @ b / ((np.linalg.norm(a)+eps) * (np.linalg.norm(b)+eps)))
```

## 35.6 JSON pretty print

```py
import json
print(json.dumps(obj, ensure_ascii=False, indent=2))
```

## 35.7 Quick HTTP (requests)

```py
import requests
r = requests.get("https://example.com", timeout=10)
r.status_code
r.text[:200]
```

## 35.8 Retry loop (basic)

```py
import time

for attempt in range(3):
    try:
        do_work()
        break
    except Exception:
        time.sleep(1)
```

---

# 36) Mini “Symbols Dictionary”

* `[]` list
* `()` tuple / call / grouping
* `{}` dict or set literal
* `:` slicing / dict key-value / type hints in def
* `*` unpack list/tuple, varargs
* `**` power / kwargs unpack
* `@decorator` wrapper
* `->` return type hint
* `:=` walrus operator (assignment expression)
* `...` ellipsis (placeholder)
* `r"..."` raw string (regex/path)

Walrus:

```py
if (n := len(s)) > 10:
    print(n)
```



