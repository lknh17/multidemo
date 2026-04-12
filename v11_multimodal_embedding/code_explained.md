# v11 代码解释
## FAISSRetriever
封装 FAISS 索引，支持 Flat (精确) 和 IVF (近似) 两种。IndexFlatIP 对归一化向量等价于余弦相似度。不安装 FAISS 时 fallback 到 numpy。

## 检索评估
Recall@K 衡量"能否找到"，NDCG 衡量"排序是否好"，MRR 关注"多快能找到第一个正确结果"。
