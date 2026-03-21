
## 1. 问题

有一个方程（线性回归模型）： $\hat{y} = \beta_{0} + \beta_{1}x$

我原计划通过两次精准的实验，然后算出两个模型参数 beta_0, beta_1

但是，由于各种原因，实验的数据非常不准，因此，直接使用两次实验结果来计算准确的模型参数变得不可能，因此，我想进行120次实验，

然后通过最小二乘法来找一个最优解，并使用 **正规方程** 直接求解以及 **随机梯度** 下降两种方式。

## 2. 正规方程

#### 数学

根据模型方程，设：X = $\begin{pmatrix} 1 & x_{0} \\ 1 & x_{1} \end{pmatrix}$ ，$\beta = \begin{pmatrix} \beta_{0} \\ \beta_{1} \end{pmatrix}$

则有，$y = X\beta$

此方程2个未知数，120个方程即120行，即 X 为 （120 x 2）的矩阵，是个超定方程，没有精确解

回忆下超定方程几个特点：

1. 没有精确解
2. Xbeta 是 X 的 列向量的线性组合
3. 由于没有精确解，y 不在 X 的列空间里
4. 我们期望找到一个 最优解y*，使得 y到y* 的距离最短，即 y*= Xbeta*
5. 这个距离我们称为残差 r - y - y*

我们使用 欧几里得范数平方来评估这个残差  即 r = ||（Xbeta* - y）|| ^2

根据最小二乘法，残差向量垂直于 X 的列空间， 即 残差 r 垂直于 X 的每一个列，即：

- X_1 ^T * r = 0
- X_2 ^T * r = 0

所以，X^T * r = 0，即 X^T(Xbeta* - y) = 0 ---> X^T X Beta* = X^T y ---> Beta* = (X^T X)^-1 X^T y

#### 代码

```r

set.seed(42)

# ----- 1. 生成模拟数据 -----
n <- 200
true_beta0 <- 2.5
true_beta1 <- 1.8

x <- runif(n, 0, 10)
noise <- rnorm(n, 0, 2)
y <- true_beta0 + true_beta1 * x + noise

cat("真实参数: β0 =", true_beta0, ", β1 =", true_beta1, "\n")
cat("==========================================\n\n")

# ----- 2. 构建设计矩阵 -----
X <- cbind(1, x)
cat("设计矩阵 X 的维度:", dim(X), "\n")
cat("X 的条件数:", kappa(X), "\n")
cat("X'X 的条件数:", kappa(t(X) %*% X), "\n\n")

# ----- 3. 正规方程求解 -----
# 公式: β = (X'X)^(-1) X'y
cat("=== 方法1: 正规方程 ===\n")

XtX <- t(X) %*% X      # X'X
Xty <- t(X) %*% y      # X'y
beta_normal <- solve(XtX) %*% Xty  # (X'X)^(-1) X'y

cat("β0 =", beta_normal[1], "\n")
cat("β1 =", beta_normal[2], "\n")

# 计算残差和MSE
y_pred_normal <- X %*% beta_normal
mse_normal <- mean((y - y_pred_normal)^2)
cat("MSE =", mse_normal, "\n\n")
```


## 3. 随机梯度下降的数学

#### 数学

假设一开始我们设置 beta_0 = (0,0) 学习率 = 0.01 批量大小为 10

我们迭代执行以下步骤：

1. 从实验获取的数据中随机选择 10 条
2. 计算这个批次数据的残差 的范数 r = yi - xi^T beta
3. 计算梯度：
	1. 对于一个批量随机梯度下降，损失函数是均方误差：L(beta) = 1/批量大小 * (y  - Xbeta)^T (y - Xbeta) 
	2.  损失函数梯度求解（对beta求偏导）：
		1. 基本公式1：偏导beta（beta ^T a) = a
		2. 基本公式2：偏导beta (beta^T A beta) = 2Abeta (A对称矩阵)
		3. 展开损失函数：L = 1/批量大小 （y^T y - 2beta^T X^Ty + beta^T X^T X beta)
		4. 对beta求导得到梯度：梯度 = 1/批量大小 (0 - 2X^T y + 2X^T X beta) = -2X^T/批量大小 (y - Xbeta)
		

#### 代码

```r
set.seed(42)

# ----- 1. 生成模拟数据 -----
n <- 200
true_beta0 <- 2.5
true_beta1 <- 1.8

x <- runif(n, 0, 10)
noise <- rnorm(n, 0, 2)
y <- true_beta0 + true_beta1 * x + noise


# ----- 5. SGD求解 -----
cat("=== 方法3: Mini-batch SGD ===\n")

learning_rate <- 0.01
batch_size <- 16
n_epochs <- 100

beta_sgd <- c(0, 0)  # 初始化 [β0, β1]
loss_history <- numeric()

for (epoch in 1:n_epochs) {
  # 打乱数据
  idx <- sample(1:n)
  X_shuffled <- X[idx, ]
  y_shuffled <- y[idx]
  
  for (i in seq(1, n, batch_size)) {
    # 取batch
    batch_idx <- i:min(i + batch_size - 1, n)
    X_batch <- X_shuffled[batch_idx, , drop = FALSE]
    y_batch <- y_shuffled[batch_idx]
    
    # 计算梯度: -X'(y - Xβ) / batch_size
    residuals <- y_batch - X_batch %*% beta_sgd
    gradient <- -t(X_batch) %*% residuals / length(batch_idx)
    
    # 更新参数
    beta_sgd <- beta_sgd - learning_rate * gradient
  }
  
  # 记录损失
  loss_history[epoch] <- mean((y - X %*% beta_sgd)^2)
}

cat("β0 =", beta_sgd[1], "\n")
cat("β1 =", beta_sgd[2], "\n")
cat("MSE =", loss_history[n_epochs], "\n\n")


# ----- 7. 可视化 -----
par(mfrow = c(1, 2))

# 图1: 拟合结果对比
plot(x, y, col = "gray60", pch = 16, cex = 0.8,
     main = "两种方法拟合对比",
     xlab = "x", ylab = "y")
abline(a = true_beta0, b = true_beta1, col = "green", lwd = 2, lty = 2)
abline(a = beta_normal[1], b = beta_normal[2], col = "blue", lwd = 2)
abline(a = beta_sgd[1], b = beta_sgd[2], col = "red", lwd = 2, lty = 3)
legend("topleft",
       legend = c("真实", "正规方程", "SGD"),
       col = c("green", "blue", "red"),
       lty = c(2, 1, 3), lwd = 2, cex = 0.8)

# 图2: SGD收敛曲线
plot(1:n_epochs, loss_history, type = "l", col = "red", lwd = 2,
     main = "SGD损失收敛",
     xlab = "Epoch", ylab = "MSE")
abline(h = mse_normal, col = "blue", lty = 2, lwd = 2)
legend("topright",
       legend = c("SGD", "正规方程解"),
       col = c("red", "blue"),
       lty = c(1, 2), lwd = 2, cex = 0.8)

par(mfrow = c(1, 1))
```



方法对比

```r
# ----- 6. 结果汇总 -----
cat("==========================================\n")
cat("结果对比:\n")
cat("==========================================\n")
results <- data.frame(
  方法 = c("真实值", "正规方程", "SGD"),
  β0 = c(true_beta0, beta_normal[1],  beta_sgd[1]),
  β1 = c(true_beta1, beta_normal[2], beta_sgd[2])
)
print(results, row.names = FALSE)
```


## 4. 扩展：从线性到真正的非线性

上面我们用正规方程和 SGD 两条路求解了线性回归。一个自然的想法是：如果模型不是直线，比如加个平方项，是不是正规方程就用不了了？

### 4.1 常见误区：多项式回归仍然是"线性"模型

考虑模型：

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2
$$

虽然对 $x$ 来说是二次曲线，但对参数 $\beta_0, \beta_1, \beta_2$ 来说仍然是**线性**的。

只需令 $z_1 = x,\; z_2 = x^2$，设计矩阵变成：

$$
X = \begin{pmatrix} 1 & x_1 & x_1^2 \\ 1 & x_2 & x_2^2 \\ \vdots & \vdots & \vdots \end{pmatrix}_{n \times 3}, \quad \boldsymbol{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{pmatrix}
$$

模型仍然是 $y = X\beta$，正规方程 $\beta^* = (X^TX)^{-1}X^Ty$ **照常使用**。

::: {.callout-important title="「线性回归」中的「线性」指的是什么？"}
指的是**对参数线性**，不是对 \(x\) 线性。只要模型能写成 $y = X\beta$（参数的线性组合），无论 \(x\) 做了什么变换（平方、取对数、交叉项……），都属于线性模型，正规方程都能用。
:::

#### 快速验证

```r
# 多项式数据：对 x 非线性，但对 β 线性
set.seed(42)
n_poly <- 200
x_poly <- runif(n_poly, 0, 10)
y_poly <- 1 + 0.5 * x_poly - 0.08 * x_poly^2 + rnorm(n_poly, 0, 1)

# 设计矩阵 [1, x, x^2] —— 仍然是 Xβ 形式！
X_poly <- cbind(1, x_poly, x_poly^2)

# 正规方程直接求解
beta_poly <- solve(t(X_poly) %*% X_poly) %*% t(X_poly) %*% y_poly

cat("多项式回归（正规方程求解）:\n")
cat("  β0 =", round(beta_poly[1], 4), " (真实: 1)\n")
cat("  β1 =", round(beta_poly[2], 4), " (真实: 0.5)\n")
cat("  β2 =", round(beta_poly[3], 4), " (真实: -0.08)\n")
cat("  MSE =", round(mean((y_poly - X_poly %*% beta_poly)^2), 4), "\n")
````

```r
#| fig-width: 7
#| fig-height: 5
#| fig-cap: "多项式回归：虽然是曲线，但正规方程照样能解"

x_seq_poly <- seq(0, 10, length.out = 200)
y_true_poly <- 1 + 0.5 * x_seq_poly - 0.08 * x_seq_poly^2
y_fit_poly <- beta_poly[1] + beta_poly[2] * x_seq_poly + beta_poly[3] * x_seq_poly^2

plot(x_poly, y_poly, col = "grey60", pch = 16, cex = 0.7,
     main = "多项式回归（正规方程）", xlab = "x", ylab = "y")
lines(x_seq_poly, y_true_poly, col = "#2ecc71", lwd = 2.5, lty = 2)
lines(x_seq_poly, y_fit_poly, col = "#3498db", lwd = 2.5)
legend("topright",
       legend = c("真实曲线", "正规方程拟合"),
       col = c("#2ecc71", "#3498db"),
       lty = c(2, 1), lwd = 2.5, cex = 0.9, bg = "white")
```

**结论**：加平方项、加立方项、加交叉项……只要模型是 $y = X\beta$ 形式，正规方程永远管用。

---

### 4.2 真正的非线性模型：饱和曲线

那什么情况下正规方程彻底失效？**当参数出现在非线性位置时。**

考虑一个在酶动力学、药物吸收、学习曲线中非常常见的**饱和模型**（Michaelis-Menten 方程）：$$\hat{y} = \frac{\beta_0 \, x}{\beta_1 + x}$$

- $\beta_0$​：**饱和值**（当 $x \to \infty$ 时 $\hat{y}$​ 趋近的最大值）
- $\beta_1$​：**半饱和常数**（$\hat{y}$​ 达到$\beta_0 / 2$ 时对应的 xxx 值）

```r
#| fig-width: 7
#| fig-height: 5
#| fig-cap: "饱和曲线：x 增大时 y 趋向一个极限值"

x_demo_nl <- seq(0, 15, length.out = 300)
y_demo_nl <- 10 * x_demo_nl / (2 + x_demo_nl)

plot(x_demo_nl, y_demo_nl, type = "l", lwd = 2.5, col = "#e74c3c",
     main = expression("饱和模型: " ~ hat(y) == frac(beta[0]~x, beta[1] + x)),
     xlab = "x", ylab = expression(hat(y)),
     ylim = c(0, 12))
abline(h = 10, lty = 2, col = "grey50")
text(14, 10.5, expression(beta[0] == 10 ~ "(饱和值)"), cex = 0.9, col = "grey40")
segments(2, 0, 2, 5, lty = 3, col = "#3498db")
segments(0, 5, 2, 5, lty = 3, col = "#3498db")
points(2, 5, pch = 16, col = "#3498db", cex = 1.5)
text(3.5, 4.5, expression(beta[1] == 2 ~ "(半饱和常数)"), cex = 0.9, col = "#3498db")
```

**为什么这是真正的非线性？** $\beta_1$​ 出现在分母中，与 xxx 相加。无论怎么做变量替换，都**不可能**把这个模型写成 $y = X\beta$ 的矩阵乘法形式。

---

### 4.3 为什么正规方程失效

回忆线性模型中正规方程的推导逻辑：

模型: $\hat{y} = X\beta \quad\Longrightarrow\quad \text{梯度: } -\frac{2}{m}X^T(y - X\beta) = 0 \quad\Longrightarrow\quad X^TX\beta = X^Ty$

最后一步是关于 $\beta$ 的**线性方程**，所以能直接解。

对于非线性模型 $\hat{y}_i = f(x_i; \beta)$

$$\text{梯度: } \frac{\partial L}{\partial \beta_j} = -\frac{2}{m}\sum_{i=1}^{m}(y_i - f(x_i;\beta)) \cdot \frac{\partial f}{\partial \beta_j} = 0$$

令梯度为零后得到的方程是**非线性**的——因为 $f$ 和 $\frac{\partial f}{\partial \beta_j}$​ 本身都依赖 $\beta$。没有办法像正规方程那样一步解出 $\beta^*$。

::: {.callout-note title="直觉理解"}  
在线性模型中，损失函数 $L(\beta)$ 是 $\beta$ 的**二次函数**（碗状曲面），碗只有一个最低点，设导数为零就能找到它。

在非线性模型中，损失函数的形状复杂得多，可能有多个山谷和鞍点，没有公式能直接跳到最低点，只能从某个起始位置出发，沿着下降方向一步步走——这正是 SGD 做的事。  
:::

**结论：只能靠迭代优化（如 SGD）逐步逼近。**

---

### 4.4 用链式法则推导梯度

损失函数仍然是均方误差，但预测值不再是简单的矩阵乘法：

$$L(\beta) = \frac{1}{m}\sum_{i=1}^{m}\left(y_i - \frac{\beta_0 x_i}{\beta_1 + x_i}\right)^2$$

用**链式法则**对每个参数求偏导：

$$
\frac{\partial L}{\partial \beta_j} = \frac{-2}{m}\sum_{i=1}^{m}\underbrace{(y_i - \hat{y}_i)}_{\text{残差 } r_i} \cdot \underbrace{\frac{\partial f}{\partial \beta_j}}_{\text{模型对参数的偏导}}
$$

分别计算模型函数 $f = \frac{\beta_0 x}{\beta_1 + x}$ 对两个参数的偏导：

$$
\frac{\partial f}{\partial \beta_0} = \frac{x}{\beta_1 + x}
$$

$$
\frac{\partial f}{\partial \beta_1} = -\frac{\beta_0 x}{(\beta_1 + x)^2}
$$

（第二个使用了商法则：分子 $\beta_0 x$ 对 $\beta_1$ 的导数为 0，分母 $\beta_1 + x$对 $\beta_1$ 的导数为 1）

所以完整的梯度为：

$$
\frac{\partial L}{\partial \beta_0} = \frac{-2}{m}\sum_{i=1}^{m} r_i \cdot \frac{x_i}{\beta_1 + x_i}
$$

$$
\frac{\partial L}{\partial \beta_1} = \frac{-2}{m}\sum_{i=1}^{m} r_i \cdot \left(-\frac{\beta_0 x_i}{(\beta_1 + x_i)^2}\right) = \frac{2}{m}\sum_{i=1}^{m} r_i \cdot \frac{\beta_0 x_i}{(\beta_1 + x_i)^2}
$$

::: {.callout-tip title="对比线性模型的梯度"}

线性模型的梯度是 $-\frac{2}{m}X^T(y - X\beta)$，其中 \(X\) 是**固定的**设计矩阵，不依赖 $\beta$。

非线性模型的梯度中，"等效的 $X$"（即偏导数矩阵）**随 $\beta$ 变化**。这意味着每走一步，梯度的计算方式都在变，正是这一点让问题无法用一个线性方程一步解决。
:::

---

### 4.5 SGD 求解非线性模型

```r
set.seed(42)

# ===== 1. 生成饱和曲线数据 =====
n <- 200
true_b0 <- 10     # 饱和值
true_b1 <- 2      # 半饱和常数

x_nl <- runif(n, 0.5, 15)
noise_nl <- rnorm(n, 0, 0.5)
y_nl <- true_b0 * x_nl / (true_b1 + x_nl) + noise_nl

cat("真实参数: β0 =", true_b0, "(饱和值), β1 =", true_b1, "(半饱和常数)\n")
cat("模型: y = β0·x / (β1 + x)\n\n")

# ===== 2. 用线性模型强行拟合（作为对照） =====
X_lin <- cbind(1, x_nl)
beta_lin <- solve(t(X_lin) %*% X_lin) %*% t(X_lin) %*% y_nl
mse_lin <- mean((y_nl - X_lin %*% beta_lin)^2)

cat("=== 对照: 线性拟合 (正规方程) ===\n")
cat("β0 =", round(beta_lin[1], 4), ", β1 =", round(beta_lin[2], 4), "\n")
cat("MSE =", round(mse_lin, 4), "\n\n")

# ===== 3. SGD 求解非线性模型 =====
learning_rate_nl <- 0.02
batch_size_nl <- 16
n_epochs_nl <- 300

beta_nl <- c(5, 5)   # 初始猜测（远离真值）
loss_history_nl <- numeric(n_epochs_nl)
beta_history_nl <- matrix(NA, nrow = n_epochs_nl, ncol = 2)

for (epoch in 1:n_epochs_nl) {
  # 打乱数据
  idx <- sample(1:n)
  x_shuf <- x_nl[idx]
  y_shuf <- y_nl[idx]

  for (i in seq(1, n, batch_size_nl)) {
    # 取 mini-batch
    batch_end <- min(i + batch_size_nl - 1, n)
    bi <- i:batch_end
    xb <- x_shuf[bi]
    yb <- y_shuf[bi]
    m  <- length(bi)

    # --- 前向传播：计算预测值 ---
    y_hat <- beta_nl[1] * xb / (beta_nl[2] + xb)

    # --- 计算残差 ---
    r <- yb - y_hat

    # --- 模型对参数的偏导数 ---
    df_db0 <- xb / (beta_nl[2] + xb)                       # ∂f/∂β0
    df_db1 <- -beta_nl[1] * xb / (beta_nl[2] + xb)^2      # ∂f/∂β1

    # --- 损失函数的梯度 ---
    grad <- c(
      -2 / m * sum(r * df_db0),    # ∂L/∂β0
      -2 / m * sum(r * df_db1)     # ∂L/∂β1
    )

    # --- 参数更新 ---
    beta_nl <- beta_nl - learning_rate_nl * grad
  }

  # 记录每个 epoch 结束时的状态
  y_pred_all <- beta_nl[1] * x_nl / (beta_nl[2] + x_nl)
  loss_history_nl[epoch] <- mean((y_nl - y_pred_all)^2)
  beta_history_nl[epoch, ] <- beta_nl
}

mse_nl <- loss_history_nl[n_epochs_nl]

cat("=== 非线性 SGD 求解 ===\n")
cat("学习率:", learning_rate_nl, " 批量大小:", batch_size_nl, " 轮数:", n_epochs_nl, "\n")
cat("β0 =", round(beta_nl[1], 4), " (真实:", true_b0, ")\n")
cat("β1 =", round(beta_nl[2], 4), " (真实:", true_b1, ")\n")
cat("MSE =", round(mse_nl, 4), "\n")
```

---

### 4.6 可视化

```r
#| fig-width: 12
#| fig-height: 10
#| fig-cap: "非线性模型：SGD 拟合 vs 线性强行拟合"

par(mfrow = c(2, 2), mar = c(4.5, 4.5, 3, 1))

# ===== 图1: 数据与拟合曲线 =====
x_curve <- seq(0.5, 15, length.out = 300)

plot(x_nl, y_nl, col = "grey60", pch = 16, cex = 0.7,
     main = "拟合结果对比", xlab = "x", ylab = "y")

# 真实饱和曲线
lines(x_curve, true_b0 * x_curve / (true_b1 + x_curve),
      col = "#2ecc71", lwd = 2.5, lty = 2)

# SGD 非线性拟合
lines(x_curve, beta_nl[1] * x_curve / (beta_nl[2] + x_curve),
      col = "#e74c3c", lwd = 2.5)

# 线性拟合（对照）
abline(a = beta_lin[1], b = beta_lin[2], col = "#3498db", lwd = 2, lty = 3)

legend("bottomright",
       legend = c("真实饱和曲线", "SGD 非线性拟合", "线性拟合 (正规方程)"),
       col = c("#2ecc71", "#e74c3c", "#3498db"),
       lty = c(2, 1, 3), lwd = 2.5, cex = 0.8, bg = "white")

# ===== 图2: SGD 损失收敛 =====
plot(1:n_epochs_nl, loss_history_nl, type = "l", col = "#e74c3c", lwd = 2,
     main = "SGD 损失收敛（非线性模型）",
     xlab = "Epoch", ylab = "MSE")
abline(h = mse_lin, col = "#3498db", lty = 3, lwd = 1.5)
abline(h = mse_nl, col = "#e74c3c", lty = 2, lwd = 1)
legend("topright",
       legend = c("非线性 SGD 轨迹", "线性模型 MSE"),
       col = c("#e74c3c", "#3498db"),
       lty = c(1, 3), lwd = 2, cex = 0.8, bg = "white")

# ===== 图3: 参数迭代轨迹 =====
plot(beta_history_nl[, 1], beta_history_nl[, 2],
     type = "l", col = adjustcolor("#e74c3c", 0.4), lwd = 1.5,
     main = expression("参数迭代轨迹（" * beta[0] * "-" * beta[1] * " 平面）"),
     xlab = expression(beta[0] ~ "(饱和值)"),
     ylab = expression(beta[1] ~ "(半饱和常数)"),
     xlim = range(c(5, beta_history_nl[, 1], true_b0)) * c(0.9, 1.05),
     ylim = range(c(true_b1, beta_history_nl[, 2], 5)) * c(0.8, 1.1))

points(5, 5, pch = 17, col = "black", cex = 1.5)
text(5, 5, " 起点 (5, 5)", pos = 4, cex = 0.8)

points(beta_nl[1], beta_nl[2], pch = 16, col = "#e74c3c", cex = 1.8)
text(beta_nl[1], beta_nl[2], "SGD 终点 ", pos = 2, cex = 0.8, col = "#e74c3c")

points(true_b0, true_b1, pch = 8, col = "#2ecc71", cex = 2)
text(true_b0, true_b1, " 真实值 (10, 2)", pos = 4, cex = 0.8, col = "#2ecc71")

# ===== 图4: 残差对比 =====
res_lin <- y_nl - X_lin %*% beta_lin
res_nl  <- y_nl - beta_nl[1] * x_nl / (beta_nl[2] + x_nl)

plot(x_nl, res_lin, col = adjustcolor("#3498db", 0.5), pch = 16, cex = 0.7,
     main = "残差对比", xlab = "x", ylab = "残差",
     ylim = range(c(res_lin, res_nl)) * 1.1)
points(x_nl, res_nl, col = adjustcolor("#e74c3c", 0.5), pch = 16, cex = 0.7)
abline(h = 0, lty = 2, col = "grey40")
legend("topleft",
       legend = c("线性模型残差", "非线性模型残差"),
       col = c("#3498db", "#e74c3c"),
       pch = 16, cex = 0.8, bg = "white")

par(mfrow = c(1, 1))
```

::: {.callout-tip title="观察残差图（右下角）"}  
线性模型的残差呈现明显的**弓形模式**（先负后正再负），说明模型结构不匹配——它在试图用直线拟合一条曲线。非线性模型的残差则随机分散在零线两侧，说明模型结构选对了。残差图是判断模型是否合适的重要工具。  
:::

---

### 4.7 结果汇总

```r
cat("==========================================\n")
cat("  非线性模型结果对比\n")
cat("==========================================\n\n")

results_nl <- data.frame(
  参数 = c("β0 (饱和值)", "β1 (半饱和常数)"),
  真实值 = c(true_b0, true_b1),
  SGD估计 = round(c(beta_nl[1], beta_nl[2]), 4)
)
print(results_nl, row.names = FALSE)

cat("\n模型选择对比:\n")
cat("------------------------------------------\n")
model_cmp <- data.frame(
  模型 = c("线性 (用正规方程)", "非线性 (用SGD)"),
  MSE = round(c(mse_lin, mse_nl), 4)
)
print(model_cmp, row.names = FALSE)
cat("\n非线性模型的 MSE 应显著低于线性模型，\n")
cat("因为数据本身就是非线性生成的。\n")
```

---

## 5. 全文知识串联

整篇笔记的逻辑链条如下：

```r
#| echo: false
#| results: asis

cat("| 概念 | 角色 | 出现位置 |\n")
cat("|:---|:---|:---|\n")
cat("| 线性回归 | 起点模型 | §1-3 |\n")
cat("| 超定方程 | 200 方程 2 未知数，无精确解 | §2 |\n")
cat("| 列空间与投影 | 最优解 = y 在列空间的投影 | §2 |\n")
cat("| 最小二乘法 | 最小化残差范数平方 | §2 |\n")
cat("| 正规方程 | 解析解（一步到位） | §2 |\n")
cat("| 损失函数 (MSE) | 将误差量化为可微标量 | §3, §4 |\n")
cat("| 矩阵求导 | 线性梯度推导 | §3 |\n")
cat("| 批量 (mini-batch) | 平衡效率与稳定性 | §3, §4 |\n")
cat("| SGD | 迭代逼近（当解析解可用时的替代方案） | §3 |\n")
cat("| 对参数的线性性 | 决定能否用正规方程的关键判据 | §4.1 |\n")
cat("| 链式法则 | 非线性模型梯度推导的核心工具 | §4.4 |\n")
cat("| 非线性最小二乘 | 只能迭代，正规方程彻底失效 | §4.2-4.6 |\n")
cat("| 残差分析 | 判断模型结构是否合适 | §4.6 |\n")
```

总结成一条线：

> 实验有噪声 → 多做实验取最优 → **超定方程** → **最小二乘** → **正规方程**（解析解）vs **SGD**（迭代解）→ 模型变非线性 → 正规方程**失效** → SGD 成为**唯一选择** → 用**链式法则**手动求梯度

::: {.callout-note title="SGD 不是唯一的迭代方法"}  
对于非线性最小二乘问题，除了 SGD，还有 Gauss-Newton、Levenberg-Marquardt 等专门的优化算法（R 中 `nls()` 函数就用了这类方法）。但 SGD 是最通用的——当模型变得更复杂（比如神经网络有数百万参数）时，SGD 及其变种（Adam、RMSProp 等）几乎是唯一可行的选择。从这个角度看，本文的饱和曲线模型虽然简单，但它走的正是通向深度学习的同一条路。  
:::


几个设计要点说明：

**关于误区纠正**：你原先的想法是"加平方项就不能用正规方程"，这其实是个很常见的混淆。§4.1 专门用一个小例子澄清了这一点——"线性"指的是对参数线性，不是对 \(x\) 线性。这个辨析本身就是一个很好的知识点。

**关于非线性模型的选择**：我没有用指数模型 $y = \beta_0 e^{\beta_1 x}$，因为指数函数在 SGD 中容易数值爆炸（$\beta_1$) 稍微偏大，梯度就会呈指数级增长，导致发散）。饱和曲线模型的梯度天然有界（分母的平方保证了这一点），用普通 SGD 就能稳定收敛，更适合教学演示。

**残差图**：右下角的残差对比图是个亮点。线性模型的残差会呈现系统性的弓形模式，直观展示"用错模型"的后果。这个图在实际数据分析中也是判断模型是否合适的标准工具。
