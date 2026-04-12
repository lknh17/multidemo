# V19 - 层级标签理解：代码详解

## 1. 分类学树构建（taxonomy.py）

### 1.1 TaxonomyTree 数据结构

```python
class TaxonomyTree:
    """分类学树：维护标签间的父子、祖孙关系"""
    
    def __init__(self, num_labels_per_level):
        # 构建层级映射
        # level 0: [0, ..., 9]     行业
        # level 1: [10, ..., 59]   品类
        # level 2: [60, ..., 259]  子品类
        self.parent = {}   # child_id → parent_id
        self.children = {} # parent_id → [child_ids]
        self.level = {}    # label_id → level
```

**为什么需要这个结构**：层级分类需要知道哪些子类属于哪个父类，才能做条件 Softmax。

### 1.2 节点遍历与查询

```python
def get_ancestors(self, node_id):
    """获取从当前节点到根的完整路径"""
    ancestors = []
    current = node_id
    while current in self.parent:
        current = self.parent[current]
        ancestors.append(current)
    return ancestors  # [parent, grandparent, ..., root]

def get_path(self, node_id):
    """获取从根到当前节点的路径"""
    return list(reversed(self.get_ancestors(node_id))) + [node_id]
```

## 2. 层级 Softmax（taxonomy.py）

### 2.1 核心实现

```python
class HierarchicalSoftmax(nn.Module):
    def forward(self, features, target_paths=None):
        log_prob = 0
        for level in range(self.num_levels):
            # 获取当前层的父节点
            parent = target_paths[:, level]
            
            # 获取每个父节点的子节点集合
            children_ids = self.tree.children[parent]
            
            # 条件 Softmax：P(child | parent, x)
            logits = self.classifiers[level](features)  # [B, num_children]
            log_prob += F.log_softmax(logits, dim=-1)
```

**关键理解**：每一层只在该层的兄弟节点间做 Softmax，而不是在所有类别间。这大大减少计算量（从 200 类变成最多 50 类的 Softmax）。

### 2.2 层级权重

```python
# 细粒度层权重更大：λ_k = (k+1) / Σ(k+1)
weights = [(k + 1) for k in range(depth)]
weights = [w / sum(weights) for w in weights]
# depth=3: [1/6, 2/6, 3/6] = [0.167, 0.333, 0.5]
```

## 3. 标签传播 GNN（taxonomy.py）

### 3.1 图构建

```python
class LabelPropagationGNN(nn.Module):
    def __init__(self, tree, d_model, n_layers=2):
        # 从树结构构建邻接矩阵
        edges = []
        for child, parent in tree.parent.items():
            edges.append([child, parent])
            edges.append([parent, child])  # 双向边
        
        # 对称归一化
        # Â = D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.diag(degree.pow(-0.5))
        self.adj_norm = D_inv_sqrt @ A @ D_inv_sqrt
```

### 3.2 消息传递

```python
def forward(self, node_features):
    """图神经网络消息传递"""
    h = node_features
    for layer in self.layers:
        # 邻居聚合
        h_agg = torch.matmul(self.adj_norm, h)
        # 线性变换 + 激活
        h = layer(h_agg)
        h = F.relu(h)
    return h
```

**为什么有效**：通过 GNN 消息传递，父节点的信息会传播到子节点，子节点的信息也会聚合到父节点，实现层级信息的双向流动。

## 4. 双曲空间嵌入（taxonomy.py）

### 4.1 Poincaré 距离

```python
def poincare_distance(u, v, eps=1e-5):
    """Poincaré 球上的测地距离"""
    diff = u - v
    norm_sq_diff = torch.sum(diff ** 2, dim=-1)
    norm_sq_u = torch.sum(u ** 2, dim=-1).clamp(max=1 - eps)
    norm_sq_v = torch.sum(v ** 2, dim=-1).clamp(max=1 - eps)
    
    # d(u,v) = arcosh(1 + 2 * ||u-v||² / ((1-||u||²)(1-||v||²)))
    arg = 1 + 2 * norm_sq_diff / ((1 - norm_sq_u) * (1 - norm_sq_v))
    return torch.acosh(arg.clamp(min=1 + eps))
```

### 4.2 指数映射

```python
def exp_map_zero(v, c=1.0, eps=1e-5):
    """从原点出发的指数映射：欧氏向量 → Poincaré 球上的点"""
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    
    # tanh(√c · ||v||) / (√c · ||v||) · v
    return torch.tanh(c ** 0.5 * v_norm) * v / (c ** 0.5 * v_norm)
```

**关键理解**：
- 根节点（粗粒度）嵌入在靠近原点的位置（范数小）
- 叶节点（细粒度）嵌入在靠近边界的位置（范数大）
- 这完美匹配树的几何结构：越靠近叶子，"空间"越大，容纳更多细粒度类别

### 4.3 投影回球内

```python
def project_to_ball(x, c=1.0, eps=1e-5):
    """确保点在 Poincaré 球内部"""
    max_norm = (1 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True)
    return x * (max_norm / norm).clamp(max=1.0)
```

## 5. 层级分类器（model.py）

### 5.1 三级级联分类

```python
class HierarchicalClassifier(nn.Module):
    """coarse → mid → fine 三级级联分类"""
    
    def forward(self, images):
        features = self.encoder(images)  # 视觉特征
        
        # Level 0: 粗粒度分类
        coarse_logits = self.coarse_head(features)
        coarse_pred = coarse_logits.argmax(-1)
        
        # Level 1: 中粒度（条件在粗粒度上）
        coarse_embed = self.coarse_embed(coarse_pred)
        mid_features = features + coarse_embed
        mid_logits = self.mid_head(mid_features)
        
        # Level 2: 细粒度（条件在中粒度上）
        mid_embed = self.mid_embed(mid_pred)
        fine_features = features + mid_embed
        fine_logits = self.fine_head(fine_features)
```

**条件分类的好处**：细粒度分类器知道"这是什么行业"后，只需在该行业下的子品类中选择，搜索空间大大缩小。

### 5.2 层级约束多标签模型

```python
class ConstrainedMultiLabelModel(nn.Module):
    def compute_consistency_loss(self, predictions):
        """层级一致性损失：子节点概率 ≤ 父节点概率"""
        loss = 0
        for child_id, parent_id in self.tree.parent.items():
            # max(0, P(child) - P(parent))
            violation = F.relu(predictions[:, child_id] - predictions[:, parent_id])
            loss += violation.mean()
        return loss
```

## 6. 标签嵌入模型（model.py）

```python
class LabelEmbeddingModel(nn.Module):
    """视觉-标签联合嵌入"""
    
    def forward(self, images, label_ids):
        # 视觉特征 → 嵌入空间
        visual_embed = self.visual_proj(self.encoder(images))
        
        # 标签 → 嵌入空间（双曲或欧氏）
        label_embed = self.label_embed(label_ids)
        if self.use_hyperbolic:
            label_embed = exp_map_zero(label_embed)
            visual_embed = exp_map_zero(visual_embed)
        
        # 对比损失
        distance = poincare_distance(visual_embed, label_embed)
```

**联合嵌入的价值**：将视觉特征和标签映射到同一空间，使得图像自然"靠近"其正确标签，实现零样本泛化。
