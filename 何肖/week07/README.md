# 第 7 周作业 — 序列标注（NER）
姓名：何肖

基于 **peoples_daily** 人民日报数据集，对比五种命名实体识别方案：BERT 序列标注（Linear / CRF）、大模型 API（zero-shot / few-shot）、本地 LLM SFT（LoRA）。

## 任务说明

- **数据集**：`data/peoples_daily/`（train / validation / test，BIO 标注）
- **实体类型**：PER（人名）、ORG（机构）、LOC（地名）
- **目标**：训练并评估多种 NER 方案，对比精度、标签合法性与适用场景

## 五种方案

| 方案 | 脚本 | 说明 |
|------|------|------|
| BERT + Linear | `src/train.py` / `src/evaluate.py` | BERT + 线性分类头，逐 token 预测 BIO |
| BERT + CRF | `src/train.py --use_crf` | 在线性头上加 CRF，解码时约束 BIO 转移 |
| LLM zero-shot | `src_llm/llm_ner.py` | DeepSeek API，仅任务描述 |
| LLM few-shot | `src_llm/llm_ner.py` | DeepSeek API，附加 3 个标注样例 |
| Qwen2 SFT | `src_llm/train_sft.py` + `evaluate_sft.py` | Qwen2-0.5B-Instruct + LoRA 微调 |

## 目录结构

```
作业/
├── README.md              # 本说明
├── RESULT.MD              # 运行日志与结果分析（详细）
├── data/peoples_daily/    # 数据集
├── src/                   # BERT 方案
│   ├── train.py           # 训练
│   ├── evaluate.py        # 评估
│   ├── model.py           # BertNER / BertCRFNER
│   ├── dataset.py         # 数据加载与 BIO 处理
│   ├── explore_data.py    # 数据探索
│   └── compare_results.py # 五方案汇总对比
├── src_llm/               # LLM 方案
│   ├── llm_ner.py         # API zero/few-shot 评估
│   ├── train_sft.py       # LoRA 微调
│   └── evaluate_sft.py    # SFT 评估
└── outputs/
    ├── checkpoints/       # BERT 权重（best_linear.pt / best_crf.pt）
    ├── sft_adapter/       # LoRA adapter
    └── logs/              # 训练与评估 JSON 日志
```

## 环境依赖

```bash
conda activate py312   # 推荐；LLM/SFT 需 transformers、peft、openai

pip install torch transformers seqeval tqdm peft openai
```

**预训练模型**（需提前放在 `八斗/pretrain_models/`）：

| 模型 | 路径 |
|------|------|
| bert-base-chinese | `pretrain_models/bert-base-chinese` |
| Qwen2-0.5B-Instruct | `pretrain_models/Qwen2-0.5B-Instruct` |

**LLM API**（仅 `llm_ner.py` 需要）：

```bash
export DEEPSEEK_API_KEY=你的密钥
```

## 运行方式

### 1. BERT 训练与评估

```bash
cd src

# 训练 Linear 头（默认 3 epoch）
python train.py

# 训练 CRF 头
python train.py --use_crf

# 评估（validation 全量，seqeval entity F1）
python evaluate.py
python evaluate.py --use_crf
```

### 2. LLM API 评估

```bash
cd src_llm
python llm_ner.py              # 100 条分层采样，zero + few-shot
```

### 3. Qwen2 SFT

```bash
cd src_llm

# 训练（LoRA，默认 3 epoch；CPU 较慢，可先用子集试跑）
python train_sft.py
python train_sft.py --num_train 2000 --epochs 2

# 评估
python evaluate_sft.py
python evaluate_sft.py --demo   # 仅 5 条快速查看
```

### 4. 汇总对比

```bash
cd src
python compare_results.py
```

需先完成上述评估，日志写入 `outputs/logs/` 后再运行。

## 运行结果（摘要）

| 方案 | F1 | 备注 |
|------|-----|------|
| BERT + Linear | 0.9516 | validation 全量 seqeval |
| BERT + CRF | **0.9542** | 非法 BIO 序列 32 条（Linear 47 条） |
| LLM zero-shot | 0.7063 | 100 条 span F1，DeepSeek API |
| LLM few-shot | 0.7233 | 100 条 span F1，3 个 few-shot 样例 |
| Qwen2 SFT (LoRA) | 0.1142 | 100 条 span F1，仅训 1 epoch |

> **指标说明**：BERT 为 validation 全量 **seqeval entity F1**；LLM / SFT 为 validation 分层采样 100 条 **span F1**（seed=42）。跨方案 F1 为近似参考，详见 `RESULT.MD`。

## 主要结论

1. **BERT + CRF 最优**：F1 最高，CRF 转移矩阵减少非法 BIO 序列。
2. **LLM API 可用但不及 BERT**：few-shot F1 ~0.72，无需训练，适合快速验证。
3. **小模型 SFT 需充分训练**：Qwen2-0.5B 仅 1 epoch 时 F1 很低，输出格式不稳定。

完整日志、逐类型 F1 与详细分析见 **[RESULT.MD](./RESULT.MD)**。
