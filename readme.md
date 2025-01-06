## 例子1：用EM算法求解最大似然估计（MLE）

### 问题描述

我们有如下不完整的数据：

- 观察到的数据：
  \[ \mathbf{x} = [1, 2, 3, 8, 9, 10] \]
- 假设数据来自两个正态分布：
  \[ \mathcal{N}(\mu_1, \sigma_1^2) \] 和 \[ \mathcal{N}(\mu_2, \sigma_2^2) \]。
- 未知参数：
  \[ \theta = (\mu_1, \mu_2, \sigma_1^2, \sigma_2^2) \]
- 假设样本属于两个分布的类别 \[ z_i \] 是隐变量：
  \[ z_i = \{0, 1\} \]，表示样本属于 \( \mathcal{N}(\mu_1, \sigma_1^2) \) 或 \( \mathcal{N}(\mu_2, \sigma_2^2) \)。

### 目标

最大化对数似然函数：
\[
\log P(\mathbf{x} | \theta) = \sum_{i=1}^n \log \left( P(x_i | z_i=0, \theta) P(z_i=0) + P(x_i | z_i=1, \theta) P(z_i=1) \right)
\]

### EM算法

#### 初始化参数

\[ \mu_1^{(0)} = 2.0, \mu_2^{(0)} = 9.0, \sigma_1^{2(0)} = 1.0, \sigma_2^{2(0)} = 1.0 \]

#### E步

计算每个数据点属于 \( \mathcal{N}(\mu_1, \sigma_1^2) \) 或 \( \mathcal{N}(\mu_2, \sigma_2^2) \) 的概率：
\[
\gamma_{i,1} = P(z_i=0 | x_i, \theta^{(t)}) = \frac{P(x_i | \mathcal{N}(\mu_1, \sigma_1^2)) P(z_i=0)}{P(x_i | \mathcal{N}(\mu_1, \sigma_1^2)) P(z_i=0) + P(x_i | \mathcal{N}(\mu_2, \sigma_2^2)) P(z_i=1)}
\]

#### M步

根据 \( \gamma_{i,1} \) 更新参数：

- 均值：
  \[
  \mu_1^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,1} x_i}{\sum_{i=1}^n \gamma_{i,1}}, \quad \mu_2^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,2} x_i}{\sum_{i=1}^n \gamma_{i,2}}
  \]

- 方差：
  \[
  \sigma_1^{2(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,1} (x_i - \mu_1^{(t+1)})^2}{\sum_{i=1}^n \gamma_{i,1}}, \quad \sigma_2^{2(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,2} (x_i - \mu_2^{(t+1)})^2}{\sum_{i=1}^n \gamma_{i,2}}
  \]

#### 数值结果

通过若干次迭代，最终估计参数收敛到：
\[
\mu_1 \approx 2.0, \mu_2 \approx 9.0, \sigma_1^2 \approx 1.0, \sigma_2^2 \approx 1.0
\]

---

## 例子2：用EM算法求解最大后验估计（MAP）

### 问题描述

- 观察到的数据：
  \[ \mathbf{x} = [1.5, 2.0, 2.5, 8.0, 8.5, 9.0] \]
- 数据来源于两个高斯分布：
  \[ \mathcal{N}(\mu_1, \sigma_1^2), \mathcal{N}(\mu_2, \sigma_2^2) \]
- 未知参数：
  \[ \theta = (\mu_1, \mu_2, \sigma_1^2, \sigma_2^2) \]
- **先验信息**：
  \[ P(\mu_1) = \mathcal{N}(3, 1), \quad P(\mu_2) = \mathcal{N}(7, 1) \]

### 目标

最大化后验概率：
\[
P(\theta | \mathbf{x}) \propto P(\mathbf{x} | \theta) \cdot P(\theta)
\]

### EM算法

#### 初始化参数

\[ \mu_1^{(0)} = 2.0, \mu_2^{(0)} = 8.0, \sigma_1^{2(0)} = 1.0, \sigma_2^{2(0)} = 1.0 \]

#### E步

计算每个数据点属于 \( \mathcal{N}(\mu_1, \sigma_1^2) \) 或 \( \mathcal{N}(\mu_2, \sigma_2^2) \) 的概率：
\[
\gamma_{i,1} = \frac{P(x_i | \mathcal{N}(\mu_1, \sigma_1^2)) P(z_i=0)}{P(x_i | \mathcal{N}(\mu_1, \sigma_1^2)) P(z_i=0) + P(x_i | \mathcal{N}(\mu_2, \sigma_2^2)) P(z_i=1)}
\]

#### M步

根据 \( \gamma_{i,1} \) 更新参数，考虑先验：

1. 更新均值：
   \[
   \mu_1^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,1} x_i + \frac{3}{1}}{\sum_{i=1}^n \gamma_{i,1} + \frac{1}{1}}, \quad \mu_2^{(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,2} x_i + \frac{7}{1}}{\sum_{i=1}^n \gamma_{i,2} + \frac{1}{1}}
   \]

2. 更新方差：
   \[
   \sigma_1^{2(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,1} (x_i - \mu_1^{(t+1)})^2}{\sum_{i=1}^n \gamma_{i,1}}, \quad \sigma_2^{2(t+1)} = \frac{\sum_{i=1}^n \gamma_{i,2} (x_i - \mu_2^{(t+1)})^2}{\sum_{i=1}^n \gamma_{i,2}}
   \]

#### 数值结果

通过若干次迭代，最终估计参数收敛到：
\[
\mu_1 \approx 2.3, \mu_2 \approx 8.1, \sigma_1^2 \approx 0.2, \sigma_2^2 \approx 0.55
\]
