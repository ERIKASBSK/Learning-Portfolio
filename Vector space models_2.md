# Note 1 Principal Component Analysis

## 1) Core

PCA (**Principal Component Analysis**) reduces a high-dimensional (高次元) vector space into a low-dimensional (低次元) space (often 2D) so it can visualize points on an XY plot.  
It finds new axes called **principal components (主成分)** that keep as much information (variance) as possible.

Imagine:
* Spill a handful of beans on a table: they form an elongated cloud.
* The direction where the beans are **most spread out / longest** is **PC1 (first principal component)**   
* And **PCA (Principal Component Analysis)** is the method that finds it.
* The direction **perpendicular (orthogonal)** to PC1, with the next-biggest spread, is **PC2 (second principal component)**.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/ffc31f43-a621-44eb-b093-6d7e5a74231f" />

- **PC1**: direction with **maximum variance** after projection  
- **PC2**: best direction **perpendicular** to PC1

## 2) Math part

###  Mean + Centering


- Without centering first, PCA can be skewed by how far the whole dataset sits from the origin. After centering, it focuses on the data’s shape—its spread and main directions.
- 中心化しないと、PCAはデータ全体が原点からどれだけ離れているかの影響を受けやすい。中心化すると、データの形（ばらつき方・広がる方向）だけを見る。

```math
\mu=\frac{1}{n}\sum_{i=1}^{n} x_i,\quad X_c = X - \mathbf{1}\mu^T
````
* **$(X)$**: the original data matrix (each row is one data point) 元のデータ行列
* **$(X_c)$**: the centered data matrix 中心化（平均を引いた）後のデータ行列
* **$(\mathbf{1}\mu^T)$**: the mean vector $(\mu)$ repeated $(n)$ times as identical rows, so each row of $(X)$ can subtract $(\mu)$
* 平均ベクトル $(\mu)$ を **n回** 同じ行として並べたもの $(X)$ の各行から $(\mu)$ を引けるようにするため）
```PY
import numpy as np

X = np.array([[1,1],[2,1],[3,2],[4,3],[5,3]], dtype=float)  
mu = X.mean(axis=0)      
Xc = X - mu              
```
### Covariance (共分散行列)

- The covariance matrix C compresses the data’s variance—how much it spreads and in which directions—into one matrix. PCA then extracts the most important directions from it.
- 共分散行列 C は、データの「どの方向に」「どれくらい」ばらつくか（分散）を1つの行列にまとめたもの。PCAはそこから最も重要な方向を取り出す。
  
```math
C=\frac{1}{n}X_c^T X_c
```
```PY
n = Xc.shape[0]
C = (Xc.T @ Xc) / n       # d×d

```
### PCs (eigenvectors) (固有ベクトル)

```math
C v_i=\lambda_i v_i
```
```PY
eigvals, eigvecs = np.linalg.eigh(C)   
i = 0
vi = eigvecs[:, i]
lam = eigvals[i]

np.allclose(C @ vi, lam * vi)        
```
* **PC1** = $(v_1) (largest (\lambda))$
* **PC2** = $(v_2) (2nd largest (\lambda), orthogonal to (v_1))$

### Project to 2D (射影)

-Reduce high-dimensional vectors to 2D so they can be plotted and clusters can be seen
```math
W=[v_1\ v_2],\quad Z=X_c W
```

```PY
W = np.column_stack((v1, v2))
Z = Xc @ W
```

---

## 3) Numeric sample 

```math
X=\{(1,1),(2,1),(3,2),(4,3),(5,3)\},\quad n=5
````


Compute each coordinate mean

```math
\mu_x=\frac{1+2+3+4+5}{5}=3,\quad
\mu_y=\frac{1+1+2+3+3}{5}=2
```

```math
\mu=(\mu_x,\mu_y)=(3,2)
```

---

## Centered data ($X_c$)

```math
X_c = X-\mu=\{(-2,-1),(-1,-1),(0,0),(1,1),(2,1)\}
```

---

## Covariance ($C$)

Use the “column sums” form (equivalent to  $(C=\frac{1}{n}X_c^T X_c))$


```math
x=[-2,-1,0,1,2],\quad y=[-1,-1,0,1,1]
```

Compute:

```math
\sum x^2=4+1+0+1+4=10,\quad \frac{1}{n}\sum x^2=\frac{10}{5}=2.0
```

```math
\sum y^2=1+1+0+1+1=4,\quad \frac{1}{n}\sum y^2=\frac{4}{5}=0.8
```

```math
\sum xy=(2)+(1)+(0)+(1)+(2)=6,\quad \frac{1}{n}\sum xy=\frac{6}{5}=1.2
```

So:

```math
C=
\begin{bmatrix}
\frac{1}{n}\sum x^2 & \frac{1}{n}\sum xy\\
\frac{1}{n}\sum xy & \frac{1}{n}\sum y^2
\end{bmatrix}
=
\begin{bmatrix}
2.0 & 1.2\\
1.2 & 0.8
\end{bmatrix}
```

---
Given the covariance matrix:

```math
C=\begin{bmatrix}2.0 & 1.2\\ 1.2 & 0.8\end{bmatrix}
````

$PC1$ and $PC2$ are the **eigenvectors** of $C$.

---

Eigenvalues $(\lambda)$

For a $2\times2$ symmetric matrix:

```math
\begin{bmatrix}a & b\\ b & d\end{bmatrix}
```

Eigenvalues:

```math
\lambda=\frac{(a+d)\pm\sqrt{(a-d)^2+4b^2}}{2}
```

Here $a=2.0,\ b=1.2,\ d=0.8$:

```math
\lambda=\frac{2.8\pm\sqrt{(1.2)^2+4(1.2)^2}}{2}
=\frac{2.8\pm\sqrt{8.64}}{2}
```

So:

```math
\lambda_1\approx 2.7416407865,\quad \lambda_2\approx 0.0583592135
```

---

Eigenvectors $(v=(v_1,v_2))$

Definition:

```math
Cv=\lambda v
```

Use the first row of $(C-\lambda I)v=0$:

```math
(a-\lambda)v_1 + b v_2 = 0
\quad\Rightarrow\quad
\frac{v_2}{v_1}=\frac{\lambda-a}{b}
```

### $PC1$ (use $\lambda_1$)

```math
\frac{v_2}{v_1}=\frac{2.7416407865-2.0}{1.2}=0.6180339887
```

Direction ratio:

```math
v \propto (1,\ 0.6180339887)
```

Normalize to unit length:

```math
PC1\approx(0.85065081,\ 0.52573111)
```

### $PC2$ (use $\lambda_2$)

```math
\frac{v_2}{v_1}=\frac{0.0583592135-2.0}{1.2}=-1.6180339887
```

Direction ratio:

```math
v \propto (1,\ -1.6180339887)
```

Normalize (overall sign can flip):

```math
PC2\approx(-0.52573111,\ 0.85065081)
```

---

**Summary:** compute $\lambda$, use $\frac{v_2}{v_1}=\frac{\lambda-a}{b}$, then normalize $\Rightarrow$ $PC1/PC2$.


```math
PC1\approx(0.85065081,\ 0.52573111)
```

```math
PC2\approx(-0.52573111,\ 0.85065081)
```

---

## Projection results

```math
z_1=x_c\cdot PC1,\quad z_2=x_c\cdot PC2
```

| original | centered $(x_c)$ | $(z_1) (PC1)$ | $(z_2) (PC2)$ |
| -------- | -------------- | ----------: | ----------: |
| (1,1)    | (-2,-1)        | -2.22703273 |  0.20081141 |
| (2,1)    | (-1,-1)        | -1.37638192 | -0.32491970 |
| (3,2)    | (0,0)          |  0.00000000 |  0.00000000 |
| (4,3)    | (1,1)          |  1.37638192 |  0.32491970 |
| (5,3)    | (2,1)          |  2.22703273 | -0.20081141 |

MY MOOD:  
<img width="702" height="683" alt="image" src="https://github.com/user-attachments/assets/c72900bf-b8ca-4475-8018-d1291ce07967" />


