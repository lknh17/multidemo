# V22 - 评估体系与 A/B 测试：代码详解

## 1. 检索评估指标（metrics.py）

### 1.1 Recall@K

```python
class RetrievalMetrics:
    def recall_at_k(self, scores, relevance, k):
        # scores: [N] 预测分数
        # relevance: [N] 二值相关性标签
        topk_indices = scores.argsort(descending=True)[:k]
        relevant_in_topk = relevance[topk_indices].sum()
        total_relevant = relevance.sum()
        return relevant_in_topk / max(total_relevant, 1)
```

### 1.2 NDCG@K

```python
def ndcg_at_k(self, scores, relevance, k):
    # 1. 按预测分数排序
    topk_idx = scores.argsort(descending=True)[:k]
    gains = (2 ** relevance[topk_idx] - 1)

    # 2. 计算折损
    discounts = 1.0 / torch.log2(torch.arange(k) + 2)
    dcg = (gains * discounts).sum()

    # 3. 理想排序
    ideal_gains = (2 ** relevance.sort(descending=True).values[:k] - 1)
    idcg = (ideal_gains * discounts[:len(ideal_gains)]).sum()

    return dcg / max(idcg, 1e-8)
```

### 1.3 MRR

```python
def mrr(self, scores, relevance):
    ranked_idx = scores.argsort(descending=True)
    ranked_rel = relevance[ranked_idx]
    # 找第一个相关文档的位置
    first_relevant = (ranked_rel > 0).nonzero()
    if len(first_relevant) == 0:
        return 0.0
    rank = first_relevant[0].item() + 1
    return 1.0 / rank
```

### 1.4 mAP

```python
def average_precision(self, scores, relevance):
    ranked_idx = scores.argsort(descending=True)
    ranked_rel = relevance[ranked_idx]
    cum_relevant = ranked_rel.cumsum(0)
    precision_at_k = cum_relevant / torch.arange(1, len(ranked_rel) + 1)
    ap = (precision_at_k * ranked_rel).sum() / max(relevance.sum(), 1)
    return ap
```

## 2. A/B 测试框架（model.py）

### 2.1 流量分割

```python
class ABTestFramework:
    def assign_group(self, user_id):
        # 确定性哈希分组，保证同一用户始终在同一组
        hash_val = hash(str(user_id) + self.salt) % 10000
        if hash_val < self.traffic_split * 10000:
            return 'treatment'
        return 'control'
```

### 2.2 显著性检验

```python
def welch_t_test(self, control_data, treatment_data):
    n_c, n_t = len(control_data), len(treatment_data)
    mean_c, mean_t = control_data.mean(), treatment_data.mean()
    var_c, var_t = control_data.var(), treatment_data.var()

    # Welch's t 统计量
    t_stat = (mean_t - mean_c) / sqrt(var_c/n_c + var_t/n_t)

    # Welch-Satterthwaite 自由度
    nu = (var_c/n_c + var_t/n_t)**2 / (
        (var_c/n_c)**2/(n_c-1) + (var_t/n_t)**2/(n_t-1))

    p_value = 2 * (1 - t.cdf(abs(t_stat), df=nu))
    return t_stat, p_value
```

### 2.3 Bootstrap 置信区间

```python
def bootstrap_ci(self, control, treatment, n_iterations=10000):
    deltas = []
    for _ in range(n_iterations):
        # 有放回采样
        c_sample = np.random.choice(control, size=len(control))
        t_sample = np.random.choice(treatment, size=len(treatment))
        deltas.append(t_sample.mean() - c_sample.mean())

    alpha = 1 - self.confidence_level
    ci_lower = np.percentile(deltas, 100 * alpha / 2)
    ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper
```

## 3. 多臂老虎机（model.py）

### 3.1 UCB1

```python
class BanditSelector:
    def select_ucb1(self, t):
        ucb_values = []
        for k in range(self.num_arms):
            if self.counts[k] == 0:
                return k  # 未探索的臂优先
            mean_reward = self.total_rewards[k] / self.counts[k]
            bonus = self.ucb_c * sqrt(log(t) / self.counts[k])
            ucb_values.append(mean_reward + bonus)
        return np.argmax(ucb_values)
```

### 3.2 Thompson Sampling

```python
def select_thompson(self):
    samples = []
    for k in range(self.num_arms):
        # 从 Beta 后验采样
        theta = np.random.beta(self.alpha[k], self.beta[k])
        samples.append(theta)
    return np.argmax(samples)

def update_thompson(self, arm, reward):
    # 更新 Beta 后验
    self.alpha[arm] += reward
    self.beta[arm] += (1 - reward)
```

## 4. 交错实验（model.py）

```python
class InterleavingExperiment:
    def team_draft(self, list_a, list_b, k):
        merged, team_a, team_b = [], set(), set()
        ptr_a, ptr_b = 0, 0

        for i in range(k):
            # 交替选取，落后方优先
            if len(team_a) <= len(team_b):
                # A 队选
                while ptr_a < len(list_a) and list_a[ptr_a] in merged:
                    ptr_a += 1
                merged.append(list_a[ptr_a])
                team_a.add(list_a[ptr_a])
            else:
                # B 队选
                while ptr_b < len(list_b) and list_b[ptr_b] in merged:
                    ptr_b += 1
                merged.append(list_b[ptr_b])
                team_b.add(list_b[ptr_b])

        return merged, team_a, team_b
```

## 5. 公平性指标（metrics.py）

```python
class FairnessMetrics:
    def demographic_parity(self, predictions, sensitive_attr):
        group_0 = predictions[sensitive_attr == 0].float().mean()
        group_1 = predictions[sensitive_attr == 1].float().mean()
        return abs(group_0 - group_1)

    def equalized_odds(self, predictions, labels, sensitive_attr):
        gaps = []
        for y in [0, 1]:
            mask = (labels == y)
            rate_0 = predictions[mask & (sensitive_attr == 0)].float().mean()
            rate_1 = predictions[mask & (sensitive_attr == 1)].float().mean()
            gaps.append(abs(rate_0 - rate_1))
        return max(gaps)
```
