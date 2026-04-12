# V25 - 端到端广告多模态管线（Capstone）

## 核心知识点
- 端到端广告管线（End-to-End Ad Pipeline）：编码→检索→重排→过滤→投放
- 多模态统一编码：视觉 + 文本 + 音频融合
- 多阶段检索：召回（Recall）→ 精排（Precision）
- CTR 预估（DeepFM 风格）
- 多目标排序（CTR / 相关性 / 多样性 / 新鲜度）
- 内容安全级联过滤
- 在线学习与实时更新
- 本版为 V01-V24 全部知识的综合实践

## 运行方式

```bash
python train.py --mode pipeline
python train.py --mode ctr
python train.py --mode ranker
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 广告管线全配置（创意/管线/匹配/监控） |
| pipeline_modules.py | 管线核心模块：编码器/匹配器/安全过滤/质量门控 |
| model.py | E2E 管线 / CTR 预估 / 多目标排序 / 在线学习 |
| dataset.py | 合成广告创意数据集（含曝光/点击） |
| train.py | 训练脚本（pipeline / ctr / ranker 三种模式） |
| inference.py | 推理实验（延迟分解/召回分析/Pareto/在线学习） |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画（7 步） |
| code_visual.html | 代码动画（6 步） |
