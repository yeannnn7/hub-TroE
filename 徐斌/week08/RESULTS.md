# 文本匹配多方法 / 多数据集试验结果

本文档记录 `bq_corpus` 与 `lcqmc` 两个数据集上，三种文本匹配方法的对比试验结果与解读。

## 试验配置

| 参数 | 值 |
|------|-----|
| 数据集 | `bq_corpus`、`lcqmc` |
| 方法 | BiEncoder + CosineEmbeddingLoss、BiEncoder + TripletLoss、CrossEncoder + CrossEntropyLoss |
| 预训练模型 | `bert-base-chinese` |
| BERT 层数 | 4（三种方法一致） |
| 训练样本 | 各 10,000 条（`--max_train_samples 10000`） |
| Epochs | 3 |
| Batch size | 32 |
| 评估集 | validation |
| 设备 | CPU |

### 数据集规模

| 数据集 | train | validation | 正样本比例（validation） | 平均句长（字） |
|--------|-------|------------|------------------------|---------------|
| bq_corpus | 68,960 | 8,620 | 50.2% | ~26.8 |
| lcqmc | 238,766 | 8,802 | 50.0% | ~25.0 |

### 训练命令

```bash
cd src

# bq_corpus
python train_biencoder.py --data_dir ../data/bq_corpus --loss cosine --num_hidden_layers 4
python train_biencoder.py --data_dir ../data/bq_corpus --loss triplet --num_hidden_layers 4
python train_crossencoder.py --data_dir ../data/bq_corpus --num_hidden_layers 4

# lcqmc
python train_biencoder.py --data_dir ../data/lcqmc --loss cosine --num_hidden_layers 4
python train_biencoder.py --data_dir ../data/lcqmc --loss triplet --num_hidden_layers 4
python train_crossencoder.py --data_dir ../data/lcqmc --num_hidden_layers 4

# 对比
python compare_methods.py --datasets bq_corpus lcqmc
```

### 产物路径

- Checkpoint：`outputs/checkpoints/{method}_{dataset}_best.pt`
- 训练日志：`outputs/logs/{method}_{dataset}_log.json`
- 对比结果：`outputs/logs/dataset_method_comparison.json`
- 对比图表：`outputs/figures/dataset_method_comparison.png`、`method_comparison_{dataset}.png`

---

## 总览结果（validation）

### bq_corpus

| 方法 | Accuracy | F1 (weighted) | 备注 |
|------|----------|---------------|------|
| BiEncoder (Cosine) | 0.7704 | 0.7701 | AUC=0.844，threshold=0.65 |
| BiEncoder (Triplet) | 0.7589 | 0.7584 | AUC=0.830，threshold=0.61 |
| **CrossEncoder** | **0.7821** | **0.7821** | argmax 分类 |

### lcqmc

| 方法 | Accuracy | F1 (weighted) | 备注 |
|------|----------|---------------|------|
| BiEncoder (Cosine) | 0.7210 | 0.7201 | AUC=0.801，threshold=0.88 |
| BiEncoder (Triplet) | 0.7019 | 0.7012 | AUC=0.782，threshold=0.91 |
| **CrossEncoder** | **0.7688** | **0.7676** | argmax 分类 |

### 跨数据集汇总

```
数据集        Cosine F1   Triplet F1   CrossEncoder F1
bq_corpus     0.7701      0.7584       0.7821
lcqmc         0.7201      0.7012       0.7676
```

---

## 结论解读

### 1. CrossEncoder 在两个数据集上均最优

| 数据集 | CrossEncoder F1 | 相对 BiEncoder Cosine 提升 |
|--------|-----------------|---------------------------|
| bq_corpus | 0.7821 | +1.2% |
| lcqmc | 0.7676 | +4.8% |

CrossEncoder 将两句拼接后整体送入 BERT，每一层 Self-Attention 都能跨句交互，对细粒度语义差异更敏感。lcqmc 为开放域社区问答，存在大量「关键词重叠但语义不同」的难负例，交互型模型的优势更明显。

### 2. Cosine Loss 稳定优于 Triplet Loss

| 数据集 | Cosine F1 | Triplet F1 | ΔF1 |
|--------|-----------|------------|-----|
| bq_corpus | 0.7701 | 0.7584 | +1.2% |
| lcqmc | 0.7201 | 0.7012 | +1.9% |

当前 Triplet 采用离线随机负采样：三元组数量约等于正样本数（10k 训练集约 3000–5000 个三元组），负样本质量一般。Cosine 直接使用全部 10k 标注句对，优化目标更直接，效果更稳定。

### 3. BiEncoder 需阈值搜索，lcqmc 阈值明显更高

| 数据集 | Cosine 最优阈值 | Triplet 最优阈值 |
|--------|----------------|-----------------|
| bq_corpus | 0.65 | 0.61 |
| lcqmc | 0.88 | 0.91 |

lcqmc 上正负样本的余弦相似度分布重叠更大（AUC 0.801 vs bq_corpus 的 0.844），分类边界更难确定。

### 4. bq_corpus 整体优于 lcqmc

| 方法 | bq_corpus F1 | lcqmc F1 | 差距 |
|------|-------------|----------|------|
| CrossEncoder | 0.782 | 0.768 | −1.4% |
| BiEncoder Cosine | 0.770 | 0.720 | −5.0% |
| BiEncoder Triplet | 0.758 | 0.701 | −5.7% |

可能原因：

| 特征 | bq_corpus | lcqmc |
|------|-----------|-------|
| 领域 | 金融客服问答，表述规范 | 开放域社区问答，口语化 |
| 任务难度 | 同义改写为主 | 大量表面相似、语义不同的难负例 |
| 训练集规模 | 6.9 万（本次仅用 1 万） | 23.9 万（本次仅用 1 万） |

lcqmc 全量训练集远大于 bq_corpus，但仅取 1 万条时欠拟合更严重；BiEncoder 在 lcqmc 上掉分更多，说明表示型模型对难负例更敏感。

---

## 训练过程观察

### bq_corpus：稳定收敛

**BiEncoder Cosine**

| Epoch | train_loss | val_f1 | threshold |
|-------|------------|--------|-----------|
| 1 | 0.288 | 0.753 | 0.68 |
| 2 | 0.227 | 0.767 | 0.62 |
| 3 | 0.207 | **0.770** | 0.65 |

**CrossEncoder**

| Epoch | train_acc | val_f1 |
|-------|-----------|--------|
| 1 | 0.671 | 0.753 |
| 2 | 0.777 | 0.776 |
| 3 | 0.819 | **0.782** |

### lcqmc：存在过拟合

**BiEncoder Cosine**

| Epoch | train_loss | val_f1 | threshold |
|-------|------------|--------|-----------|
| 1 | 0.272 | 0.712 | 0.90 |
| 2 | 0.217 | **0.720** | 0.88 |
| 3 | 0.195 | 0.716 | 0.88 |

**CrossEncoder**

| Epoch | train_acc | val_f1 |
|-------|-----------|--------|
| 1 | 0.762 | 0.708 |
| 2 | 0.882 | **0.768** |
| 3 | 0.915 | 0.750 |

lcqmc 上 epoch 3 训练准确率继续上升，但验证 F1 回落，说明存在过拟合。最优 checkpoint 不一定在最后一个 epoch，建议引入 early stopping。

---

## 方法选型建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| 精度优先、在线逐对打分 | CrossEncoder | 两数据集 F1 最高 |
| 大规模检索 / RAG 召回 | BiEncoder + Cosine | 可预计算句向量，推理快 |
| 排序 pipeline | BiEncoder 召回 + CrossEncoder 精排 | 兼顾速度与精度 |
| Triplet Loss（当前配置） | 不推荐 | 离线随机负采样弱于 Cosine |

---

## 后续改进方向

1. **加大 lcqmc 训练样本**：如 `--max_train_samples 50000` 或全量，BiEncoder 收益可能更明显
2. **lcqmc 引入 early stopping**：CrossEncoder epoch 2 已优于 epoch 3
3. **Triplet 换难负样本挖掘**：在线 batch 内选高相似度负例
4. **统一 BERT 层数**：当前 BiEncoder 默认 12 层、CrossEncoder 默认 4 层；公平对比时可统一设置
5. **Badcase 分析**：针对 lcqmc 上 BiEncoder 与 CrossEncoder 的错误样本做分类统计

---

## 原始数据

完整数值见 `outputs/logs/dataset_method_comparison.json`。

各方法逐 epoch 训练曲线见：

- `outputs/logs/biencoder_cosine_bq_corpus_log.json`
- `outputs/logs/biencoder_cosine_lcqmc_log.json`
- `outputs/logs/biencoder_triplet_bq_corpus_log.json`
- `outputs/logs/biencoder_triplet_lcqmc_log.json`
- `outputs/logs/crossencoder_bq_corpus_log.json`
- `outputs/logs/crossencoder_lcqmc_log.json`
