# Note 1 — DFA definition (5-tuple)
A **Deterministic Finite Automaton (DFA)** is:

$$
M = (Q,\Sigma,\delta,q_0,F)
$$

Where:

- $$Q$$ = set of states (e.g., $$\{q_0,q_1,\dots,q_5\}$$)
- $$\Sigma$$ = input alphabet (symbols allowed), e.g. $$\{a,b\}$$
- $$\delta$$ = transition function
- $$q_0$$ = start state (initial state)
- $$F$$ = set of accepting (final) states, e.g. $$\{q_4\}$$

---

# Note 2 — Determinism (what makes it “D”)
DFA is **deterministic** because:

> For every state and every input symbol, the next state is uniquely decided.

Formally:

$$
\delta: Q \times \Sigma \rightarrow Q
$$

That means: from any state, reading symbol **a** or **b**, there is **exactly one** outgoing choice.

---

# Note 3 — Transition diagram vs transition table
A DFA transition function $$\delta$$ can be written as:

### (A) Transition graph (diagram)
- nodes = states
- directed edges = transitions
- edge label = input symbol

### (B) Transition table
A DFA table is **complete**: there is **no empty cell**.

Example structure:

| State | a | b |
|------|---|---|
| q0   | q1| q5|
| q1   | q5| q2|
| ...  |...|...|

If something “doesn’t match”, DFA usually uses a **dead/sink state** (often loops to itself).

---

# Note 4 — Start state and accept states
- The **start state** $$q_0$$ is marked by a special incoming arrow (from nowhere).
- **Accepting states** are marked with a **double circle**.

A string is accepted **only if** we finish in an accepting state.

---

# Note 5 — Running a DFA on a string (extended transition function)
Single-step transition is:

$$
\delta(q,a)
$$

But for a **whole string**, we use the **extended transition function**:

$$
\delta^{*}: Q \times \Sigma^{*} \rightarrow Q
$$

Rules:

1) Empty string does nothing:

$$
\delta^{*}(q,\epsilon)=q
$$

2) For a string $$wa$$ (string $$w$$ followed by symbol $$a$$):

$$
\delta^{*}(q,wa)=\delta(\delta^{*}(q,w),a)
$$

This means: “process $$w$$ first, then process the last symbol $$a$$”.

---

# Note 6 — Acceptance condition (DFA)
A string $$w$$ is accepted by DFA iff:

$$
\delta^{*}(q_0,w)\in F
$$

So i always:
1) start at $$q_0$$  
2) follow transitions symbol-by-symbol  
3) check final state is in $$F$$  

---

# Note 7 — Example walk (idea)
If i run input string `"abba"`:

- start at $$q_0$$
- read `a` → go to some state
- read `b` → go to next
- read `b` → go to next
- read `a` → final state

If the final state is $$q_4$$ (accepting), then `"abba"` is accepted.

---

# Note 8 — Minimal Python: DFA simulator (optional but useful)
```python
def dfa_accept(dfa, w: str) -> bool:
    state = dfa["q0"]
    for ch in w:
        state = dfa["delta"][(state, ch)]
    return state in dfa["F"]

# example format
dfa = {
    "q0": "q0",
    "F": {"q4"},
    "delta": {
        ("q0","a"): "q1", ("q0","b"): "q5",
        ("q1","a"): "q5", ("q1","b"): "q2",
        # ... fill in the rest ...
    }
}

# dfa_accept(dfa, "abba")
```

---

# Note 9 — NFA concept (what changes from DFA)
A **Non-deterministic Finite Automaton (NFA)** is similar, but:

> From a state on a symbol, it may go to **multiple** possible next states.

So the transition function becomes:

$$
\delta: Q \times \Sigma \rightarrow 2^{Q}
$$

Meaning: output is a **set of states**, not a single state.

Some NFAs also allow **epsilon transitions** (move without consuming input):

$$
\delta: Q \times (\Sigma \cup \{\epsilon\}) \rightarrow 2^{Q}
$$

---

# Note 10 — What “non-deterministic” means in practice
In DFA:
- each step = one clear next state

In NFA:
- each step = **branching possibilities**
- the machine can be “in many states at once” (as a set)

So instead of tracking one state, i track a **set of active states**.

---

# Note 11 — Acceptance condition (NFA)
A string $$w$$ is accepted by NFA iff:

> **at least one path** ends in an accepting state.

Formally, using extended transition:

$$
\delta^{*}(q_0,w)\cap F \neq \emptyset
$$

So as long as the final **set of reachable states** contains something in $$F$$, it accepts.

---

# Note 12 — Minimal Python: NFA simulator with epsilon-closure (optional)
```python
def epsilon_closure(states, eps_trans):
    stack = list(states)
    closed = set(states)
    while stack:
        s = stack.pop()
        for nxt in eps_trans.get(s, set()):
            if nxt not in closed:
                closed.add(nxt)
                stack.append(nxt)
    return closed

def nfa_accept(nfa, w: str) -> bool:
    current = epsilon_closure({nfa["q0"]}, nfa.get("eps", {}))

    for ch in w:
        nxt_states = set()
        for s in current:
            nxt_states |= nfa["delta"].get((s, ch), set())
        current = epsilon_closure(nxt_states, nfa.get("eps", {}))

    return len(current & nfa["F"]) > 0

# example format
nfa = {
    "q0": "q0",
    "F": {"q4"},
    "delta": {
        ("q0","a"): {"q1","q2"},   # nondeterministic branch
        ("q1","b"): {"q3"},
    },
    "eps": {
        "q2": {"q3"}               # epsilon transition
    }
}

# nfa_accept(nfa, "ab")
```

---

# Note 13 — DFA vs NFA (clean comparison)
| Feature | DFA | NFA |
|--------|-----|-----|
| Next state per symbol | exactly 1 | 0 / 1 / many |
| Transition function | $$Q\times\Sigma\to Q$$ | $$Q\times\Sigma\to2^Q$$ |
| Epsilon moves | no | sometimes yes |
| Acceptance | end state in $$F$$ | some reachable end state in $$F$$ |
| Implementation | simple | simulate with set of states |
| Power | same class of languages (regular) | same class of languages (regular) |

Important note:
- DFA and NFA recognize the **same language class** (regular languages)
- Any NFA can be converted to an equivalent DFA (subset construction)

---

# Note 14 — One-line summary
- DFA = one clear path only  
- NFA = many possible paths, accept if any path works  
