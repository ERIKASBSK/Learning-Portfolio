## ⚖️ Conditional Statements  
（条件付き実行／條件式執行）

---

### 1️⃣ Conditional Execution（条件付き実行／條件式執行）
Conditional execution means the program can **decide whether or not to run** certain parts of the code.  
Sequential execution just runs line by line, but with conditions, the program becomes smarter — *that’s where `if` comes in.*

条件付き実行とは、**条件に応じてコードの一部を実行するかどうかを選べる**ことです。  
順次実行は単に上から下に処理を進めますが、`if` によってプログラムはより「賢く」動作します。

所謂條件式執行，就是讓程式能**依條件決定是否執行某段代碼**。  
一般情況下程式會一行一行執行，但加上 `if`，它就變得更「聰明」。

---

### 2️⃣ Indentation （インデント／縮排）

Indentation shows which lines belong to which block.  
Python uses **indentation instead of braces `{}`**, usually 4 spaces.  
Mixing tabs and spaces = chaos (and errors).

インデントは「どのコードがどの条件に属するか」を示します。  
Python は `{}` の代わりに **インデント（4 スペース）** を使います。  
タブとスペースを混ぜるとエラーの原因になります。

縮排用來表示**哪一段代碼屬於哪個條件或區塊**。  
Python 不用 `{}`，而是靠縮排（通常 4 個空格）。  
不要混用 tab 與空白，否則會報錯。

---

### 3️⃣ Comparison Operators  （比較演算子／比較運算子）

| Symbol | English | 日本語 | 中文 |
|:--:|:--|:--|:--|
| `<` | less than | より小さい | 小於 |
| `<=` | less than or equal to | 以下 | 小於或等於 |
| `==` | equal to | 等しい | 等於 |
| `>` | greater than | より大きい | 大於 |
| `>=` | greater than or equal to | 以上 | 大於或等於 |
| `!=` | not equal to | 等しくない | 不等於 |

💡 `=` is **assignment** (put value into variable).  
`==` is **comparison** (ask if they’re the same).

---

### 4️⃣ `if ... else` （もし〜なら／如果...否則）

The `else` clause runs when the `if` condition is `False`.    
if` の条件が `False` のときに `else` ブロックが実行されます。
當 `if` 條件為 `False` 時，會執行 `else` 區塊。

```python
x = 5
if x > 10:
    print("Big")
else:
    print("Small")
```

---

### 5️⃣ Multi-way Decisions — elif （複数分岐／多條件判斷）

elif = “else + if” → for multiple choices.
Python checks top to bottom; first True wins, the rest are ignored.

elif は “else + if”。複数の条件を順に評価し、最初に True になったブロックだけが実行されます。

elif 是 “else + if”，用來進行多條件判斷。
Python 從上到下檢查，第一個為 True 的條件就執行，後面都略過。

```python
if x < 2:
    print("Small")
elif x < 10:
    print("Medium")
else:
    print("Large")
```

---

### 6️⃣ Common Pitfall （ありがち失敗／常見坑）    
Putting x < 20 before x < 10 causes smaller numbers to match early and skip the rest.
一度マッチしたブロックがあれば、その下は実行されません。条件の順序に注意。    
只要上面條件成立，下面的就不會執行。寫程式要注意條件順序。    

```python
if x < 20:      # ← catches everything below 20
    ...
elif x < 10:    # ← never reached
    ...
```
---
