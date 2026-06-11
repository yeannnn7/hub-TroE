# 第 6 周作业 — 文本分类（TNEWS）

**学员**：何肖  
**任务**：中文新闻标题 15 类分类  
**评估**：validation 随机采样 **200 条**，`seed=42`

## 对比结果

| 方法 | 训练数据 | 准确率 | 无法解析 |
|------|----------|--------|----------|
| BERT fine-tune（参考） | 53,360 | **56.18%** | 0% |
| Qwen2 zero-shot | 0 | 30.00% | 43.0% |
| Qwen2 few-shot（k=2/类） | 0 | 32.50% | 38.5% |
| **Qwen2 SFT（LoRA）** | 5,000 | **58.00%** | 1.0% |

> BERT 为全量 validation 结果（`train_log_cls.json`）；LLM 为 200 条采样评估。
>
> ## 复现实验命令

```bash
conda activate py312
cd src_llm

# Zero-shot
python classify_llm.py --num_samples 200 --seed 42

# Few-shot
python classify_llm.py --few_shot 2 --num_samples 200 --seed 42

# SFT 评估（需 outputs/sft_adapter/ + Qwen2-0.5B 权重）
python evaluate_sft.py --num_samples 200 --seed 42
```

BERT 训练（参考，耗时较长）：

```bash
cd ../src
python train.py --pool cls --epochs 3 --batch_size 16
```

## 结论

- **SFT（LoRA）最优**：5K 数据 + 0.22% 可训练参数，准确率 58%，接近 BERT 全量。
- **few-shot 略优于 zero-shot**：+2.5%，但远不如 SFT。
- **生成式需关注解析**：SFT 后无法解析降至 1%；BERT 无此问题。


详细配置、思考题与复现命令见 **[results.md](./results.md)**。
