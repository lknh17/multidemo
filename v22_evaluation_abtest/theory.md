# V22 - 评估体系与 A/B 测试：数学原理

## 1. 离线检索评估指标

### 1.1 Recall@K

在 Top-K 结果中召回相关文档的比例：

$$
\text{Recall@K} = \frac{|\{\text{relevant docs in top-K}\}|}{|\{\text{all relevant docs}\}|}
$$

### 1.2 NDCG（归一化折损累积增益）

DCG 对排名靠后的结果施加对数折损：

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}
$$

IDCG 是理想排序下的 DCG（将所有相关文档按相关性排在最前）：

$$
\text{IDCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i^*} - 1}{\log_2(i + 1)}
$$

归一化：

$$
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
$$

### 1.3 MRR（平均倒数排名）

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}
$$

其中 $\text{rank}_q$ 是查询 $q$ 第一个相关结果的排名位置。

### 1.4 mAP（平均精度均值）

单个查询的 Average Precision（带插值）：

$$
\text{AP} = \sum_{k=1}^{N} P(k) \cdot \Delta r(k) = \sum_{k=1}^{N} \frac{\text{TP}(k)}{k} \cdot \text{rel}(k)
$$

$$
\text{mAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)
$$

## 2. 统计显著性检验

### 2.1 Welch's t-test

不假设两组方差相等的 t 检验：

$$
t = \frac{\bar{X}_A - \bar{X}_B}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}
$$

自由度（Welch-Satterthwaite）：

$$
\nu = \frac{\left(\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}\right)^2}{\frac{(s_A^2/n_A)^2}{n_A-1} + \frac{(s_B^2/n_B)^2}{n_B-1}}
$$

### 2.2 Bootstrap 置信区间

从观测数据有放回采样 B 次，构造差值分布：

$$
\delta^{(b)} = \bar{X}_A^{(b)} - \bar{X}_B^{(b)}, \quad b = 1, ..., B
$$

百分位置信区间：

$$
\text{CI}_{1-\alpha} = [\delta_{(\alpha/2)}, \delta_{(1-\alpha/2)}]
$$

若 CI 不包含 0，则差异显著。

### 2.3 样本量估计

给定效应量 $\delta$、显著性水平 $\alpha$、统计功效 $1-\beta$：

$$
n = \frac{(z_{\alpha/2} + z_\beta)^2 \cdot 2\sigma^2}{\delta^2}
$$

## 3. 多臂老虎机

### 3.1 Thompson Sampling

每个臂维护 Beta 后验分布（伯努利奖励）：

$$
\theta_k \sim \text{Beta}(\alpha_k, \beta_k)
$$

更新规则：
- 选中臂 $k$，观察到奖励 $r \in \{0, 1\}$：

$$
\alpha_k \leftarrow \alpha_k + r, \quad \beta_k \leftarrow \beta_k + (1 - r)
$$

### 3.2 UCB1（Upper Confidence Bound）

$$
\text{UCB}_k = \bar{X}_k + c \sqrt{\frac{\ln N}{n_k}}
$$

其中 $\bar{X}_k$ 是臂 $k$ 的平均奖励，$N$ 是总轮数，$n_k$ 是臂 $k$ 被选次数，$c$ 是探索参数。

### 3.3 Epsilon-Greedy

$$
a_t = \begin{cases}
\arg\max_k \bar{X}_k & \text{概率 } 1 - \epsilon \\
\text{Uniform}(1, K) & \text{概率 } \epsilon
\end{cases}
$$

可使用衰减探索率：$\epsilon_t = \epsilon_0 / \sqrt{t}$

## 4. 交错实验（Interleaving）

### 4.1 Team Draft Interleaving

给定两个排序列表 $L_A$, $L_B$，交替从中选取：

$$
\text{TeamDraft}(L_A, L_B) \to L_{mixed}, T_A, T_B
$$

其中 $T_A$, $T_B$ 分别是 A、B 贡献的文档集合。

用户点击统计：

$$
\text{wins}_A = |\{q : |clicks \cap T_A| > |clicks \cap T_B|\}|
$$

$$
\Delta_{AB} = \frac{\text{wins}_A - \text{wins}_B}{\text{wins}_A + \text{wins}_B + \text{ties}}
$$

### 4.2 优势

- 比传统 A/B 测试灵敏度高 10-100 倍
- 每个用户同时看到两个系统的结果
- 消除用户间差异的影响

## 5. 公平性指标

### 5.1 Demographic Parity

$$
|P(\hat{Y}=1 | A=0) - P(\hat{Y}=1 | A=1)| \leq \epsilon
$$

### 5.2 Equalized Odds

$$
P(\hat{Y}=1 | Y=y, A=0) = P(\hat{Y}=1 | Y=y, A=1), \quad \forall y \in \{0, 1\}
$$
