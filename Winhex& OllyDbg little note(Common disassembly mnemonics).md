# WinHex / OllyDbg x86 筆記：NOP、EIP 是什麼？

## WinHex
- **Hex（十六進位）**：檔案的原始 bytes  raw bytes of the file
- **Offset（位移/位置）**：每個 byte 在檔案中的座標  the position/address of a byte in the file
- **ASCII 欄**：把 bytes 當成文字硬顯示（多半是亂碼）   bytes interpreted as text (often gibberish)

---

## 1) What is NOP?
- **NOP = No Operation（不做任何事）**  
- CPU 會「執行它」，但沒有任何效果，只是往下一條走   CPU executes it, but it has no effect—just moves on

###  Common NOP byte
- **x86 單一 NOP**：`90`  


###  NOP
- **填充/對齊（padding / alignment）**  
- **編譯器留空位（reserved space）**  
  **Compiler leaves reserved space**
- You’ll often see long runs like: `90 90 90 90 ...`

---

## EIP?
- **EIP = Instruction Pointer（指令指標）**  
- 意思是：CPU「下一條要執行」的指令位址  
  It tells the CPU the address of the **next instruction** to execute

### 32-bit / 64-bit 差別 / 32-bit vs 64-bit
- **32-bit：EIP**  
- **64-bit: RIP** (same concept, larger register)

---

## 3)  Common CPU registers
> 暫存器＝CPU 內建的超高速小變數  Registers = tiny ultra-fast variables inside the CPU

| Register | 中文 | English intuition |
|---|---|---|
| EAX | 常放運算結果 | common accumulator/result |
| EBX | 一般用途 | general purpose |
| ECX | 計數/迴圈常用 | counter/loop |
| EDX | 一般用途/乘除常用 | general / mul-div |
| ESI/EDI | 資料搬移常見 | data move helpers |
| ESP | 堆疊頂端位置 | stack pointer |
| EBP | 目前函式堆疊基準 | stack base/frame pointer |
| EIP | 下一條指令位置 | next instruction pointer |

---

## 4) EFLAGS（旗標）/ EFLAGS (status flags)
> 很多跳轉（JZ/JNZ）就是看旗標  Many branches (JZ/JNZ) depend on these flags

| Flag | 中文 | English |
|---|---|---|
| ZF | 是否為 0 | Zero Flag (result == 0) |
| CF | 進位/借位 | Carry Flag |
| SF | 正負號 | Sign Flag |
| OF | 溢位 | Overflow Flag |

---

## 5) 反組譯常見指令（助記符）/ Common disassembly mnemonics

| 指令 / Instruction | 中文 | English |
|---|---|---|
| MOV | 搬資料：A = B | move data: A = B |
| PUSH / POP | 堆疊放入 / 取出 | push/pop stack |
| CALL | 呼叫函式 | call function |
| RET | 回到上一層 | return |
| CMP | 比較（影響旗標） | compare (sets flags) |
| TEST | 位元測試（影響旗標） | bit test (sets flags) |
| JMP | 無條件跳走 | unconditional jump |
| JZ / JE | 等於/為零就跳 | jump if zero/equal |
| JNZ / JNE | 不等於/非零就跳 | jump if not zero/not equal |

---

## 6) NOP 跟 EIP 常一起出現的原因 / Why NOP and EIP show up together
- **EIP 指著你「現在要跑哪」**  
  **EIP points to “where execution goes next”**
- **NOP 是「跑了也不改變任何狀態」的指令**  
  **NOP changes nothing**
- 所以除錯時會看到：EIP 往前走，途中可能踩過一堆 NOP  
  In debugging: EIP moves forward and may step through many NOPs

---
