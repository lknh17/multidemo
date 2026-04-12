"""
V19 - 分类学树与标签层级核心模块
==================================
1. TaxonomyTree：分类学树（父子遍历、路径查询）
2. HierarchicalSoftmax：层级 Softmax
3. LabelPropagationGNN：标签传播图神经网络
4. HyperbolicEmbedding：Poincaré 球双曲嵌入
5. LabelConsistencyChecker：层级一致性检查器

参考：
- Hierarchical Softmax (Morin & Bengio, 2005)
- Poincaré Embeddings (Nickel & Kiela, NIPS 2017)
- Label Propagation (Zhu & Ghahramani, 2002)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set

from config import LabelHierarchyConfig, LabelEmbeddingConfig


# ============================================================
#  分类学树
# ============================================================

class TaxonomyTree:
    """
    分类学树：维护标签间的父子、祖孙关系
    
    结构示例（3 层，[10, 50, 200]）：
    - Level 0（根）: 虚拟根节点 -1
    - Level 1（粗）: 标签 0~9       （10 个行业）
    - Level 2（中）: 标签 10~59     （50 个品类）
    - Level 3（细）: 标签 60~259    （200 个子品类）
    """

    def __init__(self, num_labels_per_level: List[int]):
        """
        Args:
            num_labels_per_level: 每层标签数量，如 [10, 50, 200]
        """
        self.num_levels = len(num_labels_per_level)
        self.num_labels_per_level = num_labels_per_level
        self.total_labels = sum(num_labels_per_level)

        # 计算每层标签的起始 ID
        self.level_offsets = [0]
        for n in num_labels_per_level:
            self.level_offsets.append(self.level_offsets[-1] + n)

        # 构建映射关系
        self.parent: Dict[int, int] = {}       # child_id → parent_id
        self.children: Dict[int, List[int]] = {}  # parent_id → [child_ids]
        self.level_of: Dict[int, int] = {}     # label_id → level

        self._build_tree()

    def _build_tree(self):
        """构建层级树结构"""
        # 为所有节点设置层级
        for lv in range(self.num_levels):
            start = self.level_offsets[lv]
            end = self.level_offsets[lv + 1]
            for i in range(start, end):
                self.level_of[i] = lv
                if i not in self.children:
                    self.children[i] = []

        # 构建父子关系（均匀分配子节点）
        for lv in range(1, self.num_levels):
            parent_start = self.level_offsets[lv - 1]
            parent_end = self.level_offsets[lv]
            child_start = self.level_offsets[lv]
            child_end = self.level_offsets[lv + 1]

            n_parents = parent_end - parent_start
            n_children = child_end - child_start
            children_per_parent = n_children // n_parents
            remainder = n_children % n_parents

            child_idx = child_start
            for p in range(parent_start, parent_end):
                n_c = children_per_parent + (1 if p - parent_start < remainder else 0)
                for _ in range(n_c):
                    self.parent[child_idx] = p
                    self.children[p].append(child_idx)
                    child_idx += 1

    def get_ancestors(self, node_id: int) -> List[int]:
        """获取从当前节点到根的祖先路径（不含自身）"""
        ancestors = []
        current = node_id
        while current in self.parent:
            current = self.parent[current]
            ancestors.append(current)
        return ancestors  # [parent, grandparent, ...]

    def get_path(self, node_id: int) -> List[int]:
        """获取从根到当前节点的完整路径（含自身）"""
        return list(reversed(self.get_ancestors(node_id))) + [node_id]

    def get_siblings(self, node_id: int) -> List[int]:
        """获取同级兄弟节点（含自身）"""
        if node_id not in self.parent:
            # 根层节点，返回同层所有节点
            lv = self.level_of[node_id]
            start = self.level_offsets[lv]
            end = self.level_offsets[lv + 1]
            return list(range(start, end))
        parent_id = self.parent[node_id]
        return self.children[parent_id]

    def get_level_labels(self, level: int) -> List[int]:
        """获取指定层的所有标签 ID"""
        start = self.level_offsets[level]
        end = self.level_offsets[level + 1]
        return list(range(start, end))

    def tree_distance(self, a: int, b: int) -> int:
        """计算两个节点在树上的最短路径长度"""
        path_a = set(self.get_path(a))
        path_b = set(self.get_path(b))
        # 最近公共祖先 = 两条路径的交集中最深的节点
        common = path_a & path_b
        if not common:
            return len(self.get_path(a)) + len(self.get_path(b))
        lca_depth = max(len(self.get_path(c)) for c in common)
        depth_a = len(self.get_path(a))
        depth_b = len(self.get_path(b))
        return (depth_a - lca_depth) + (depth_b - lca_depth)

    def local_to_global(self, level: int, local_id: int) -> int:
        """将层内局部 ID 转换为全局 ID"""
        return self.level_offsets[level] + local_id

    def global_to_local(self, global_id: int) -> Tuple[int, int]:
        """将全局 ID 转换为 (level, local_id)"""
        lv = self.level_of[global_id]
        local_id = global_id - self.level_offsets[lv]
        return lv, local_id


# ============================================================
#  层级 Softmax
# ============================================================

class HierarchicalSoftmax(nn.Module):
    """
    层级 Softmax：逐层条件分类
    
    P(leaf | x) = Π_{k=0}^{D-1} P(v_{k+1} | v_k, x)
    
    每层只在兄弟节点间做 Softmax，计算量大幅降低
    """

    def __init__(self, tree: TaxonomyTree, d_model: int, temperature: float = 1.0):
        super().__init__()
        self.tree = tree
        self.temperature = temperature
        self.num_levels = tree.num_levels

        # 每层的分类头
        self.classifiers = nn.ModuleList()
        for lv in range(tree.num_levels):
            n_classes = tree.num_labels_per_level[lv]
            self.classifiers.append(nn.Linear(d_model, n_classes))

        # 层级权重：细粒度层权重更大
        weights = [(k + 1) for k in range(tree.num_levels)]
        total = sum(weights)
        self.level_weights = [w / total for w in weights]

    def forward(self, features: torch.Tensor,
                target_levels: Optional[List[torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D] 视觉特征
            target_levels: 每层的目标标签（局部 ID），用于计算损失
        Returns:
            logits_per_level, loss（如果有 target）
        """
        results = {}
        all_logits = []

        for lv in range(self.num_levels):
            logits = self.classifiers[lv](features) / self.temperature  # [B, C_lv]
            all_logits.append(logits)

        results['logits_per_level'] = all_logits

        # 计算层级损失
        if target_levels is not None:
            loss = 0.0
            for lv in range(self.num_levels):
                lv_loss = F.cross_entropy(all_logits[lv], target_levels[lv])
                loss += self.level_weights[lv] * lv_loss
            results['loss'] = loss

        return results

    def predict(self, features: torch.Tensor) -> List[List[int]]:
        """逐层贪心预测：返回从根到叶的路径"""
        results = self.forward(features)
        paths = []
        B = features.shape[0]

        for b in range(B):
            path = []
            for lv in range(self.num_levels):
                local_id = results['logits_per_level'][lv][b].argmax().item()
                global_id = self.tree.local_to_global(lv, local_id)
                path.append(global_id)
            paths.append(path)

        return paths


# ============================================================
#  标签传播 GNN
# ============================================================

class LabelPropagationGNN(nn.Module):
    """
    基于图神经网络的标签传播
    
    在分类学树上传播标签信息：
    - 父→子：粗粒度约束细粒度
    - 子→父：细粒度信息聚合
    """

    def __init__(self, tree: TaxonomyTree, d_model: int, n_layers: int = 2):
        super().__init__()
        self.tree = tree
        self.n_nodes = tree.total_labels
        self.n_layers = n_layers

        # 构建邻接矩阵
        self.register_buffer('adj_norm', self._build_adj_matrix())

        # GNN 层
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])

    def _build_adj_matrix(self) -> torch.Tensor:
        """构建对称归一化邻接矩阵 Â = D^{-1/2} A D^{-1/2}"""
        N = self.n_nodes
        # 邻接矩阵（含自环）
        adj = torch.eye(N)
        for child, parent in self.tree.parent.items():
            if child < N and parent < N:
                adj[child, parent] = 1.0
                adj[parent, child] = 1.0

        # 对称归一化
        degree = adj.sum(dim=1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

        return adj_norm

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        图上的消息传递
        
        Args:
            node_features: [N, D] 或 [B, N, D] 节点特征
        Returns:
            传播后的节点特征，形状同输入
        """
        has_batch = node_features.dim() == 3

        if has_batch:
            B, N, D = node_features.shape
            h = node_features
            adj = self.adj_norm.unsqueeze(0).expand(B, -1, -1)
        else:
            h = node_features
            adj = self.adj_norm

        for i in range(self.n_layers):
            # 消息传递：聚合邻居
            h_agg = torch.matmul(adj, h)
            # 线性变换
            h = self.layers[i](h_agg)
            h = self.norms[i](h)
            if i < self.n_layers - 1:
                h = F.relu(h)

        return h


# ============================================================
#  双曲空间嵌入（Poincaré 球）
# ============================================================

def poincare_distance(u: torch.Tensor, v: torch.Tensor,
                      eps: float = 1e-5) -> torch.Tensor:
    """
    Poincaré 球上的测地距离
    
    d(u,v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
    """
    diff_norm_sq = torch.sum((u - v) ** 2, dim=-1)
    u_norm_sq = torch.sum(u ** 2, dim=-1).clamp(max=1 - eps)
    v_norm_sq = torch.sum(v ** 2, dim=-1).clamp(max=1 - eps)

    arg = 1 + 2 * diff_norm_sq / ((1 - u_norm_sq) * (1 - v_norm_sq) + eps)
    return torch.acosh(arg.clamp(min=1 + eps))


def exp_map_zero(v: torch.Tensor, c: float = 1.0,
                 eps: float = 1e-5) -> torch.Tensor:
    """
    从原点出发的指数映射：欧氏向量 → Poincaré 球上的点
    
    exp_0(v) = tanh(√c · ||v||) · v / (√c · ||v||)
    """
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(c ** 0.5 * v_norm) * v / (c ** 0.5 * v_norm)


def project_to_ball(x: torch.Tensor, c: float = 1.0,
                    eps: float = 1e-5) -> torch.Tensor:
    """确保点在 Poincaré 球内部（范数 < 1/√c）"""
    max_norm = (1 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


class HyperbolicEmbedding(nn.Module):
    """
    双曲空间标签嵌入
    
    利用 Poincaré 球模型表示层级标签：
    - 粗粒度标签 → 靠近原点（范数小）
    - 细粒度标签 → 靠近边界（范数大）
    - 父子关系 → 双曲距离小
    """

    def __init__(self, config: LabelEmbeddingConfig):
        super().__init__()
        self.config = config
        self.curvature = config.curvature

        # 欧氏空间中的嵌入（之后映射到双曲空间）
        self.embed = nn.Embedding(config.label_vocab_size, config.embedding_dim)

        # 初始化：小范数，确保映射后在球内
        nn.init.uniform_(self.embed.weight, -0.01, 0.01)

    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            label_ids: [B] 或 [B, L] 标签 ID
        Returns:
            Poincaré 球上的嵌入向量
        """
        e = self.embed(label_ids)
        # 指数映射到双曲空间
        h = exp_map_zero(e, self.curvature)
        # 投影确保在球内
        h = project_to_ball(h, self.curvature)
        return h

    def compute_distance_matrix(self, label_ids: torch.Tensor) -> torch.Tensor:
        """计算标签间的双曲距离矩阵"""
        embeddings = self.forward(label_ids)  # [N, D]
        N = embeddings.shape[0]
        # 两两距离
        u = embeddings.unsqueeze(1).expand(-1, N, -1)
        v = embeddings.unsqueeze(0).expand(N, -1, -1)
        return poincare_distance(u, v)


# ============================================================
#  层级一致性检查器
# ============================================================

class LabelConsistencyChecker:
    """
    检查预测结果是否满足层级一致性：
    - 子节点为正 → 父节点也必须为正
    - 父节点为负 → 子节点也必须为负
    """

    def __init__(self, tree: TaxonomyTree):
        self.tree = tree

    def check_consistency(self, predictions: torch.Tensor,
                          threshold: float = 0.5) -> Dict[str, float]:
        """
        Args:
            predictions: [B, C] 多标签预测概率
            threshold: 二值化阈值
        Returns:
            一致性统计指标
        """
        binary = (predictions > threshold).float()
        B, C = binary.shape

        violations = 0
        total_pairs = 0

        for child, parent in self.tree.parent.items():
            if child < C and parent < C:
                # 子节点为正但父节点为负 = 违反
                child_pos = binary[:, child]
                parent_neg = 1 - binary[:, parent]
                violations += (child_pos * parent_neg).sum().item()
                total_pairs += B

        consistency_rate = 1.0 - violations / max(total_pairs, 1)

        return {
            'consistency_rate': consistency_rate,
            'violations': int(violations),
            'total_pairs': total_pairs,
        }

    def enforce_consistency(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        强制执行层级一致性（自底向上传播）
        
        如果子节点概率 > threshold，则将父节点概率提升为
        max(parent_prob, child_prob)
        """
        result = predictions.clone()
        C = result.shape[1]

        # 自底向上：从最细粒度开始
        for lv in range(self.tree.num_levels - 1, 0, -1):
            for label_id in self.tree.get_level_labels(lv):
                if label_id < C and label_id in self.tree.parent:
                    parent_id = self.tree.parent[label_id]
                    if parent_id < C:
                        result[:, parent_id] = torch.max(
                            result[:, parent_id], result[:, label_id]
                        )

        return result
