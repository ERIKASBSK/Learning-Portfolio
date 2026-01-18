
# Note 1 — Theory of Computation = what computers can do

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/957c7a5c-1d32-4469-8844-84f5a3ad2c2b" />

Theory of computation includes:

- **Automata theory**
- **Formal languages + grammars**
- **Computability**
- **Complexity**

Quick brain split:

- Automata + grammars + computability = **what is possible in principle**
- Complexity = **what is practical / efficient**

---

# Note 3 — What is an automaton (my simple definition)
An **automaton** is an abstract computer model that can:

- accept input
- produce output
- maybe use temporary storage
- make decisions (based on current state + input)

So it’s like “a simplified computer skeleton”.

---

# Note 5 — Abstract model of a computer (4 memory parts)
This video explains computer memory like 4 parts:

- **Input memory**: stores input from user
- **Output memory**: stores final output
- **Program memory**: stores the program (instructions)
- **Temporary memory**: stores intermediate results (scratchpad)

Example: compute **x^3**
- input x = 2
- temp compute x^2 = 4
- temp compute x^3 = 8
- output = 8

So in abstract view:
- CPU + program memory = “automaton core”
- temporary memory decides how powerful it is

---

# Note 6 — Key classification idea: memory = power
Automata are classified mainly by **what temporary memory they have**:

## (1) Finite Automata (FA)
- **no temporary memory**
- limited but super fast/simple

## (2) Pushdown Automata (PDA)
- has a **stack** as temporary memory
- stack operations: **push / pop**
- medium power

## (3) Turing Machine (TM)
- has **random access memory** (or an unbounded tape idea)
- strongest model
- basically “anything a real computer can compute”

So power order:

FA  ->  PDA  ->  TM  
(weak)      (strong)

---

# Note 7 — Finite Automata (FA) = “state machine vibe”
FA = states + transitions  
It reads input symbol by symbol.

It can’t “remember” long history, only current state.

Practical vibe:
- vending machines
- simple text matching
- lexical analysis (tokenizing) in compilers

### Python mini-demo (FA-ish via regex)
This isn’t literally building FA, but regex is basically FA-friendly.
```python
import re

def is_integer(s: str) -> bool:
    return bool(re.fullmatch(r"[+-]?\d+", s))

print(is_integer("123"))   # True
print(is_integer("-42"))   # True
print(is_integer("12a"))   # False
```

> FA is good for patterns i can decide with “just states”, no deep memory.

---

# Note 8 — Pushdown Automata (PDA) = stack = “parentheses matching”
PDA = FA + **stack memory**

Why stack matters:
- programming languages need nesting:
  - `( ... ( ... ) ... )`
  - `{ ... { ... } ... }`

Stack operations:
- push when i see "("
- pop when i see ")"
- if i pop an empty stack -> invalid

### Python mini-demo (stack check parentheses)
```python
def ok_parentheses(s: str) -> bool:
    stack = []
    for ch in s:
        if ch == "(":
            stack.append(ch)      # push
        elif ch == ")":
            if not stack:
                return False
            stack.pop()           # pop
    return len(stack) == 0

print(ok_parentheses("(())"))   # True
print(ok_parentheses("(()"))    # False
print(ok_parentheses("())("))   # False
```

So PDA shows up in:
- parsing
- syntax analysis in compilers
- anything with “nested structure”

---

# Note 9 — Turing Machine (TM) = “general purpose computation”
TM is the most powerful automaton model.

Main idea:
> If a problem is solvable by a real computer, it’s solvable by a Turing machine.

TM has “unbounded memory” conceptually, so it can simulate any algorithm.

Practical vibe:
- general programs
- full computation (not just pattern matching)

### Python mini-demo (toy “tape” idea, not real TM)
Just to visualize “expandable memory”:
```python
tape = list("1011")  # pretend this is a tape
head = 0

# move right and write
head += 2
tape[head] = "0"

print("".join(tape))  # 1001
```

This is not a full TM, but it helps me imagine “read/write + move”.

---

# Note 10 — Comparing power (what each one can solve)
From left to right: more powerful

- **FA** solves limited pattern problems  
- **PDA** solves FA problems + nested structure problems  
- **TM** solves everything a computer can do  

So:

FA ⊂ PDA ⊂ TM

---

# Note 11 — Real applications

- **Text processing** (pattern recognition)
- **Lexical analysis** (FA-ish)
- **Syntax analysis / parsing** (PDA-ish)
- **General computation** (TM)

- tokenizing -> FA
- grammar / nesting -> PDA
- full program -> TM

---

# Note 12 — final 
Automata are “computer models”, and the biggest difference is:

**how much temporary memory they have**  
FA (none) -> PDA (stack) -> TM (unbounded/general)

That memory decides their computational power.
