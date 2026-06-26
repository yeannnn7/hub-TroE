# 姓名：何肖

# 文本匹配项目（bq_corpus）

基于银行客服问句数据集 **bq_corpus**，实现并对比多种文本匹配方案：BiEncoder、CrossEncoder、LLM API、Qwen2 SFT。

## 数据集

| 划分 | 条数 | 说明 |
|------|------|------|
| train | 68,960 | 正负约 50:50 |
| validation | 8,620 | 训练时选 checkpoint |
| test | 8,620 | 最终评估 |

数据路径：`data/bq_corpus/`

## 方法概览

| 方法 | 脚本 | 特点 |
|------|------|------|
| BiEncoder + Cosine | `src/train_biencoder.py` | 双塔编码，余弦相似度 + 阈值 |
| BiEncoder + Triplet | `src/train_biencoder.py --loss triplet` | 三元组损失 |
| CrossEncoder | `src/train_crossencoder.py` | 句对拼接，直接分类 |
| LLM API | `src_llm/llm_compare.py` | DeepSeek zero-shot |
| Qwen2 SFT | `src_llm/train_sft.py` | LoRA 微调 |

预训练模型：`bert-base-chinese`（4 层）、`Qwen2-0.5B-Instruct`

## 项目结构

```
├── data/bq_corpus/          # 数据集
├── src/                     # BERT 相关
│   ├── train_biencoder.py
│   ├── train_crossencoder.py
│   ├── evaluate.py          # 单模型详细评估
│   ├── compare_methods.py   # 三方法横向对比
│   └── analyze_badcases.py  # 错误案例分析
├── src_llm/                 # LLM 相关
│   ├── train_sft.py
│   ├── evaluate_sft.py
│   └── llm_compare.py
└── outputs/
    ├── checkpoints/         # 模型权重
    ├── logs/                # 训练/评估日志
    └── figures/             # 可视化图表
```

## 环境安装

```bash
pip install -r requirements.txt
```

预训练模型需放在 `pretrain_models/` 目录（与作业目录同级）。

## 运行流程

### 1. 数据探索

```bash
cd src
python explore_data.py
```

### 2. 训练

```bash
# BiEncoder（默认 Cosine，3 epoch）
python train_biencoder.py --loss cosine --epochs 3 --batch_size 32

# BiEncoder Triplet
python train_biencoder.py --loss triplet --epochs 3 --batch_size 32

# CrossEncoder
python train_crossencoder.py --epochs 3 --batch_size 32
```

### 3. 评估（test 集）

训练时只在 validation 上算 F1 选模型；最终成绩需单独跑 test：

```bash
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt \
  --split test

python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_triplet_best.pt \
  --split test

python evaluate.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt \
  --split test
```

### 4. 方法对比 & Badcase 分析

```bash
python compare_methods.py --split test
python analyze_badcases.py --model_type biencoder --split test
```

### 5. LLM 方案

```bash
cd ../src_llm

# API 对比（需配置 API Key）
python llm_compare.py

# SFT 训练（默认 5000 条；全量用 --num_train -1）
python train_sft.py
python train_sft.py --num_train -1 --epochs 1

# SFT 评估
python evaluate_sft.py
```

## 实验结果

### Validation 集（训练时选模型）

| 方法 | Accuracy | F1 | AUC |
|------|----------|-----|-----|
| BiEncoder Cosine | 0.8638 | 0.8636 | 0.9242 |
| BiEncoder Triplet | 0.8581 | 0.8580 | 0.9281 |
| CrossEncoder | **0.8802** | **0.8802** | **0.9483** |

### Test 集（最终成绩）

| 方法 | Accuracy | F1 | AUC |
|------|----------|-----|-----|
| BiEncoder Cosine | 0.8568 | 0.8567 | 0.9172 |
| BiEncoder Triplet | 0.8571 | 0.8571 | 0.9303 |
| CrossEncoder | **0.8787** | **0.8787** | **0.9470** |
| DeepSeek API | 0.7300 | 0.4490 | — |
| Qwen2 SFT (LoRA) | 0.6450 | 0.7300* | — |

\* SFT 评估在 validation 随机抽 200 条，F1 为正例 F1。

### 主要结论

1. **CrossEncoder > BiEncoder**：句对联合编码，精度最高，但推理慢、不可向量化。
2. **Cosine ≈ Triplet**：test 上 F1 差距 < 0.001，两种 Loss 效果接近。
3. **BERT 微调 >> LLM zero-shot/SFT**：小模型 + 领域数据微调效果最好。
4. **SFT 策略**：先用 5K 条探 3 epoch，发现 epoch 1 val_loss 最优；再用全量 68K 条训 1 epoch，val_loss 从 0.114 降至 0.062。

### Badcase 摘要（BiEncoder Cosine，test）

- 整体准确率 85.3%，错误 1267 条
- FP（误报相似）1004 条，FN（漏判相似）263 条
- 典型 FP：近义改写被判相似，如「借钱」↔「贷款」
- 典型 FN：同义但用词差异大，如「金额转出怎么操作」↔「想把微众钱转出怎么整」
- 优化方向：Hard Negative Mining、BiEncoder 召回 + CrossEncoder 精排级联

## 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/logs/biencoder_*_log.json` | 训练日志（val F1） |
| `outputs/logs/crossencoder_log.json` | 训练日志 |
| `outputs/logs/method_comparison.json` | 三方法对比 |
| `outputs/logs/sft_results.json` | SFT 评估结果 |
| `outputs/figures/biencoder_test_sim_dist.png` | 相似度分布 |
| `outputs/figures/biencoder_badcase_dist.png` | Badcase 分布 |

---

## 附录：详细运行记录

### LLM SFT 评估结果

```
=================================================================
LLM SFT 文本匹配评估结果
=================================================================
  样本数      : 200（有效: 200，parse_fail: 0）
  Accuracy    : 0.6450
  F1 (weighted): 0.6208
  F1 (正例)    : 0.7300
  均值耗时     : 0.67s/条（GPU）

多方对比（bq_corpus validation 集，所有方案均使用 Accuracy + F1，直接可比）
  ┌──────────────────────────────────────────┬──────────┬──────────┐
  │ 方法                                     │ Accuracy │ F1(pos)  │
  ├──────────────────────────────────────────┼──────────┼──────────┤
  │ BiEncoder + CosineEmbeddingLoss          │ （见日志）│ 0.8636   │
  │ BiEncoder + TripletLoss                  │ （见日志）│ 0.8580   │
  │ CrossEncoder + CrossEntropyLoss          │ （见日志）│ 0.8802   │
  │ DeepSeek API zero-shot                   │ 0.7300   │ 0.4490   │
  │ Qwen2-0.5B SFT（LoRA）                   │ 0.6450   │ 0.7300   │
  └──────────────────────────────────────────┴──────────┴──────────┘

结果已保存 → outputs/logs/sft_results.json
```

### SFT 训练探索（5K 条，3 epoch）

```
Epoch 1/3 | train_loss=0.1232 val_loss=0.1142 | 3819s
✓ 最优 LoRA adapter 已保存 (val_loss=0.1142)
Epoch 2/3 | train_loss=0.0921 val_loss=0.1168 | 3829s
Epoch 3/3 | train_loss=0.0658 val_loss=0.1149 | 3838s
```

结论：epoch 1 val_loss 最低，后续过拟合 → 改用全量 68K 条训 1 epoch（`train_sft.json` val_loss=0.062）。

### evaluate.py — BiEncoder Cosine（test）

```bash
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt \
  --split test
```

```
BiEncoder 评估结果（test，8620 条）
  最优阈值: 0.71
  Accuracy: 0.8568
  F1      : 0.8567
  AUC     : 0.9172

              precision    recall  f1-score   support

         不相似       0.87      0.83      0.85      4238
          相似       0.84      0.88      0.86      4382

    accuracy                           0.86      8620
   macro avg       0.86      0.86      0.86      8620
weighted avg       0.86      0.86      0.86      8620
```

### evaluate.py — BiEncoder Triplet（test）

```bash
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_triplet_best.pt \
  --split test
```

```
BiEncoder 评估结果（test，8620 条）
  最优阈值: 0.56
  Accuracy: 0.8571
  F1      : 0.8571
  AUC     : 0.9303

              precision    recall  f1-score   support

         不相似       0.85      0.86      0.86      4238
          相似       0.86      0.86      0.86      4382

    accuracy                           0.86      8620
   macro avg       0.86      0.86      0.86      8620
weighted avg       0.86      0.86      0.86      8620
```

### evaluate.py — CrossEncoder（test）

```bash
python evaluate.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt \
  --split test
```

```
CrossEncoder 评估结果（test，8620 条）
  Accuracy: 0.8787
  F1      : 0.8787
  AUC     : 0.9470

              precision    recall  f1-score   support

         不相似       0.88      0.88      0.88      4238
          相似       0.88      0.88      0.88      4382

    accuracy                           0.88      8620
   macro avg       0.88      0.88      0.88      8620
weighted avg       0.88      0.88      0.88      8620
```

### compare_methods.py

```bash
python compare_methods.py
```

```
Cosine vs Triplet (Δ):
  Accuracy: -0.0057  F1: -0.0056
  → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）
```

### analyze_badcases.py — BiEncoder Cosine（test）

```bash
python analyze_badcases.py --model_type biencoder --split test
```

```
整体准确率: 0.8530  错误数: 1267

============================================================
Bad Case 汇总  (共 1267 个错误)
────────────────────────────────────────────────────────────
  FP 假阳性（预测相似，实际不同）: 1004 条
    其中高置信度错误  (Δscore>0.15): 776 条
    其中临界错误     (Δscore≤0.15): 228 条
  FN 假阴性（预测不同，实际相似）:  263 条
    其中高置信度错误  (Δscore>0.15): 157 条
    其中临界错误     (Δscore≤0.15): 106 条

────────────────────────────────────────────────────────────
Bad Case 语言特征分析：

  【FP（假阳性）】共 1004 条
    长度差     : 均值=32.9  中位=4
    s1 长度    : 均值=11.3
    s2 长度    : 均值=39.2
    字符 Jaccard: 均值=0.213  （1=完全重叠，0=无共同字符）

  【FN（假阴性）】共 263 条
    长度差     : 均值=7.4  中位=5
    s1 长度    : 均值=13.2
    s2 长度    : 均值=12.7
    字符 Jaccard: 均值=0.199  （1=完全重叠，0=无共同字符）

============================================================

  FP 高置信度错误（score最高的5条） (展示 5 条)：
    score=0.999  | '我什么时候可以借钱'
                  | '什么时候可以贷款给我'

    score=0.999  | '我的微众银行为何登陆不了'
                  | '银行下载了，怎么不能登陆'

    score=0.998  | '为什么提示我未满足微从银行的审批条件？'
                  | '不符合微众银行审批要求'

    score=0.998  | '怎样去取消借款'
                  | '借款申请中如何取消借款'

    score=0.998  | '前面打电话没接到'
                  | '没接到电话'


  FP 临界错误（5条） (展示 5 条)：
    score=0.538  | '一般电话确认多久会打？'
                  | '一般多久接贷成功'

    score=0.587  | '我今天有一笔款到期，能否延迟两天'
                  | '延迟一天还款'

    score=0.571  | '可能提前还款吗'
                  | '微粒贷还款为什么不能用零钱？'

    score=0.546  | '一般多久才来电话'
                  | '再次电话要等多久'

    score=0.519  | '当天借款，当天就还款，还计息'
                  | '开通了不用会有费用产生吗'


  FN 高置信度错误（score最低的5条） (展示 5 条)：
    score=-0.175  | '想把微众钱转出怎么整'
                  | '金额转出怎么操作'

    score=-0.173  | '为什么还款日期还是第一次借的日期'
                  | '不知道还款日每一笔都一样'

    score=-0.120  | '手机号码已经更改，到无法收到验证码'
                  | '到最后验证码的下一步怎么失败的，请教我一下，'

    score=-0.088  | '老是输入验证码，怎么回事'
                  | '我怎么收不到验证码啊？'

    score=-0.087  | '还没收到款取现金需要手续费吗'
                  | '资金还没有到账'


  FN 临界错误（5条） (展示 5 条)：
    score=0.397  | '请问半年流水怎么打？'
                  | '打印交易流水'

    score=0.410  | '如何更改还款次数'
                  | '怎么更改还款时间'

    score=0.402  | '不记得密码'
                  | '以前大号密码忘记了'

    score=0.377  | '怎么样撤销贷款'
                  | '想取消借款订单'

    score=0.490  | '为什么还不来'
                  | '不扣款？？？今天还款日，不扣？'

  图表已保存 → outputs/figures/biencoder_badcase_dist.png
```

### 优化方向建议

```
============================================================
优化方向建议（基于当前 bad case 分析）
============================================================

【1】数据层面
  ├─ 难负样本增强（Hard Negative Mining）
  │    当前负样本是随机采样，FP 案例中很多是"话题相关但语义不同"的句对。
  │    → 用训练好的 BiEncoder 在大规模数据中挖掘相似度高但标签为 0 的对，
  │      加入训练集，提升负例的区分难度。
  │
  ├─ 数据增强（正样本扩充）
  │    bq_corpus 虽正负约 50:50，但同义问句表达方式有限，TripletLoss 三元组仍受限。
  │    → 对正样本做同义改写（换词、调序），扩充正例数量。
  │      可用 LLM API 批量生成改写句。
  │
  └─ 跨数据集迁移
       LCQMC / AFQMC 等数据集包含更多样的问句对。
       → 先在 LCQMC（238K 对）上预训练 Sentence-BERT，再 fine-tune 到 bq_corpus，
         利用大数据集的语义泛化能力。

【2】模型层面（FP 字符重叠不高，主要是语义理解不足）
  ├─ 增加 BERT 层数（4 → 8 → 12 层）
  │    浅层 BERT 对语义的建模能力有限，更深层能捕捉更细粒度的语义差异。
  │
  └─ 换用领域预训练模型
       MacBERT / RoBERTa-Chinese 在通用中文语料上有更好的初始化，
       bq_corpus 中很多错误源于银行客服领域术语理解不准确。

【3】训练策略层面（针对 FN：词汇重叠低但语义相同的同义句）
  ├─ SimCSE 对比学习预训练
  │    同一句话 dropout 两次得到两个正例，大 batch 内其他句子为负例。
  │    这种方式能让模型学到"用不同词说同一个意思"的不变性。
  │
  └─ 调小 TripletLoss 的 margin（0.3 → 0.1）
       如果正例本身语义就不太相似（同义但换了词），过大的 margin 反而
       要求 sim(a,p) 比 sim(a,n) 高出太多，训练信号消失。

【4】评估与部署层面
  ├─ 阈值校准
  │    当前阈值在 val 集上网格搜索，但线上真实问句分布可能与验证集不同。
  │    → 收集真实线上日志，按实际分布重新校准阈值（Platt scaling 等）。
  │
  ├─ 两阶段级联（最实用的工程改进）
  │    BiEncoder（召回 Top-K）→ CrossEncoder（精排 Top-1）
  │    → 可用当前两个 checkpoint 直接组合，无需重新训练。
  │
  └─ 训练更多 epoch + 全量 12 层
       本次演示用 4 层 × 3 epoch 快速验证，完整训练预计提升 5~10 个 F1 点。
       → 建议尝试：4 层 vs 12 层，1 epoch vs 5 epoch 的 2×2 消融。
```
