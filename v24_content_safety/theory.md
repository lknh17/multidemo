# V24 - 内容安全与合规：数学原理

## 1. 多标签安全分类

### 1.1 问题定义

一张图片/一段文本可能同时触发多个安全类别（如 NSFW + 暴力），因此是**多标签分类**问题。

每个类别独立预测：

$$
P(y_k = 1 | \mathbf{x}) = \sigma(\mathbf{w}_k^T \mathbf{f}(\mathbf{x}) + b_k)
$$

损失函数（Binary Cross-Entropy）：

$$
\mathcal{L}_{BCE} = -\sum_{k=1}^{K} [y_k \log p_k + (1 - y_k) \log(1 - p_k)]
$$

### 1.2 阈值优化（F1-optimal Threshold）

默认阈值 0.5 通常不是最优的。对每个类别 k，寻找最大化 F1 的阈值：

$$
\tau_k^* = \arg\max_{\tau} F_1(\tau) = \arg\max_{\tau} \frac{2 \cdot P(\tau) \cdot R(\tau)}{P(\tau) + R(\tau)}
$$

其中：

$$
P(\tau) = \frac{TP(\tau)}{TP(\tau) + FP(\tau)}, \quad R(\tau) = \frac{TP(\tau)}{TP(\tau) + FN(\tau)}
$$

### 1.3 ROC / AUC 分析

ROC 曲线绘制 True Positive Rate vs False Positive Rate：

$$
TPR = \frac{TP}{TP + FN}, \quad FPR = \frac{FP}{FP + TN}
$$

AUC（曲线下面积）衡量分类器整体性能，等价于：

$$
AUC = P(\hat{y}_{pos} > \hat{y}_{neg})
$$

## 2. 对抗攻击

### 2.1 FGSM（Fast Gradient Sign Method）

单步攻击，沿损失梯度方向加扰动：

$$
\mathbf{x}_{adv} = \mathbf{x} + \epsilon \cdot \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}, y))
$$

### 2.2 PGD（Projected Gradient Descent）

多步迭代攻击，每步投影回 ε-ball：

$$
\mathbf{x}^{(t+1)} = \Pi_{\mathbf{x} + \mathcal{S}} \left( \mathbf{x}^{(t)} + \alpha \cdot \text{sign}(\nabla_{\mathbf{x}} \mathcal{L}(\theta, \mathbf{x}^{(t)}, y)) \right)
$$

其中 $\Pi$ 是投影算子，$\mathcal{S} = \{\delta : ||\delta||_\infty \leq \epsilon\}$。

### 2.3 对抗训练

Min-Max 优化框架：

$$
\min_\theta \mathbb{E}_{(\mathbf{x}, y)} \left[ \max_{\delta \in \mathcal{S}} \mathcal{L}(\theta, \mathbf{x} + \delta, y) \right]
$$

内层最大化生成最强攻击，外层最小化在最强攻击下的损失。

## 3. 水印嵌入

### 3.1 DWT 域嵌入

对图像做二维离散小波变换（DWT），在中频子带（LH/HL）嵌入水印：

$$
\tilde{C}_{LH}(i,j) = C_{LH}(i,j) + \alpha \cdot w_k
$$

其中 $\alpha$ 控制嵌入强度，$w_k \in \{-1, +1\}$ 是水印比特。

### 3.2 DCT 域嵌入

将图像分块做 DCT，在中频系数上嵌入：

$$
\tilde{D}(u,v) = D(u,v) \cdot (1 + \alpha \cdot w_k)
$$

### 3.3 水印检测

提取嵌入系数，计算与原始水印的相关性：

$$
\rho = \frac{\sum_k \hat{w}_k \cdot w_k}{\sqrt{\sum_k \hat{w}_k^2} \cdot \sqrt{\sum_k w_k^2}}
$$

若 $\rho > \tau_{detect}$，判定含水印。

## 4. 毒性评分与校准

### 4.1 多模态毒性融合

文本 + 图像特征融合后评分：

$$
\mathbf{f}_{fused} = \text{CrossAttention}(\mathbf{f}_{text}, \mathbf{f}_{image})
$$

$$
s_{toxic} = \sigma(\mathbf{w}^T \mathbf{f}_{fused} + b)
$$

### 4.2 Platt Scaling 校准

模型输出的 logits 通常未校准（confident ≠ calibrated）。Platt Scaling 学习温度参数：

$$
p_{calibrated} = \sigma(a \cdot z + b)
$$

其中 $a, b$ 在验证集上通过最小化 NLL 学习：

$$
\min_{a,b} -\sum_i [y_i \log \sigma(a z_i + b) + (1-y_i) \log(1 - \sigma(a z_i + b))]
$$

### 4.3 Expected Calibration Error (ECE)

将预测概率分桶，计算每个桶的置信度与准确率之差：

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(B_m) - conf(B_m)|
$$

### 4.4 温度缩放（Temperature Scaling）

Platt Scaling 的简化版，只学一个标量温度 T：

$$
p = \sigma(z / T)
$$

$T > 1$ 使分布更平滑（降低过度自信），$T < 1$ 使分布更尖锐。

## 5. 级联安全检查

### 5.1 级联策略

快速规则过滤 → 轻量模型 → 重量级模型，逐级升级：

$$
\text{decision} = \begin{cases}
\text{reject} & \text{if rule\_check fails} \\
\text{reject} & \text{if light\_model score} > \tau_1 \\
\text{heavy\_model}(\mathbf{x}) & \text{if light\_model score} \in [\tau_0, \tau_1] \\
\text{accept} & \text{if light\_model score} < \tau_0
\end{cases}
$$

### 5.2 成本感知阈值

不同安全类别的误判成本不同（FP vs FN）：

$$
\tau_k^* = \arg\min_\tau \left[ c_{FP}^{(k)} \cdot FPR(\tau) + c_{FN}^{(k)} \cdot FNR(\tau) \right]
$$

对高风险类别（如 CSAM），$c_{FN}$ 极大 → 阈值很低 → 宁可误报也不漏报。
