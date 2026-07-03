# 基于Qwen的文本匹配模型

## 目录结构

```
llm_model/
├── dataset.py          # 数据处理模块
├── model.py            # Qwen模型定义
├── train.py            # 训练脚本
├── requirements.txt    # 依赖清单
└── README.md           # 操作说明
```

## 功能概述

基于Qwen2.5-0.5B-Instruct的文本匹配模型，具有以下特性：

- **默认采用LoRA高效微调**：大幅减少可训练参数
- **数据随机采样**：支持按比例采样数据，默认使用10%数据
- **完整的训练流程**：训练、验证、测试一体化
- **丰富的输出**：详细的训练报告和预测结果

## 模型说明

### 基础模型
- **模型名称**: Qwen2.5-0.5B-Instruct
- **模型路径**: `pretrain_models/Qwen2___5-0___5B-Instruct`
- **模型大小**: 约0.5B参数
- **任务类型**: 文本匹配（判断两个句子是否相似）

### LoRA配置
- **默认启用**: 是
- **目标模块**: q_proj, k_proj, v_proj, o_proj
- **可训练参数比例**: 约0.1-1%

## 数据集说明

BQ Corpus是一个中文银行问答数据集：
- 训练集：100,000条
- 验证集：10,000条
- 测试集：10,000条

数据格式：
```json
{"sentence1": "文本1", "sentence2": "文本2", "label": 0或1}
```

- **label=1**: 句子相似
- **label=0**: 句子不相似

## 使用方法

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本训练（默认使用10%数据+LoRA）

```bash
cd llm_model
python train.py
```

### 使用完整数据集

```bash
python train.py --sample_ratio 1.0
```

### 使用30%数据训练

```bash
python train.py --sample_ratio 0.3
```

### 自定义LoRA参数

```bash
python train.py --lora_r 16 --lora_alpha 32
```

### 禁用LoRA（全参数微调）

```bash
python train.py --use_lora False
```

## 命令行参数

### 数据参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --max_len | int | 256 | 最大序列长度 |
| --sample_ratio | float | 0.1 | 数据集随机采样比例（0.0-1.0） |
| --seed | int | 42 | 随机种子 |

### 模型参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --model_path | str | pretrain_models/Qwen2___5-0___5B-Instruct | 模型路径 |

### LoRA参数（默认启用）
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --use_lora | bool | True | 启用LoRA微调 |
| --lora_r | int | 8 | LoRA秩 |
| --lora_alpha | int | 16 | LoRA alpha参数 |
| --lora_dropout | float | 0.1 | LoRA dropout |

### 训练参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| --batch_size | int | 2 | 批处理大小 |
| --epochs | int | 1 | 训练轮数 |
| --learning_rate | float | 1e-4 | 学习率 |
| --weight_decay | float | 0.01 | 权重衰减 |
| --accumulation_steps | int | 4 | 梯度累积步数 |

## 输出文件

训练完成后在 `outputs/llm/` 目录生成：

| 文件 | 说明 |
|------|------|
| lora_adapter/ | LoRA适配器权重目录 |
| test_results.json | 测试集预测结果 |
| training_history.json | 训练历史记录 |
| config.json | 训练配置参数 |
| training_report.txt | 详细训练报告 |

## 训练流程

1. **数据加载**: 读取train.jsonl, validation.jsonl, test.jsonl
2. **数据采样**: 根据sample_ratio随机采样指定比例的数据（默认10%）
3. **模型初始化**: 加载Qwen模型，应用LoRA配置
4. **参数统计**: 显示可训练参数比例
5. **训练循环**: 使用因果语言模型训练
6. **模型验证**: 每个epoch后在验证集上评估
7. **模型保存**: 保存LoRA适配器权重
8. **测试评估**: 使用最优模型在测试集上评估
9. **结果保存**: 保存预测结果和报告

## 评估指标

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数

## 预测结果格式

测试结果保存在 `test_results.json`，每条记录包含：

```json
{
  "sentence1": "句子1",
  "sentence2": "句子2",
  "generated_text": "是",
  "prediction": 1,
  "label": 1,
  "correct": true
}
```

## LoRA优势

| 对比项 | 全参数微调 | LoRA微调 |
|--------|-----------|----------|
| 可训练参数 | 100% | ~0.1-1% |
| GPU显存占用 | 高 | 降低50%+ |
| 训练时间 | 长 | 缩短2-3倍 |
| 模型性能 | 好 | 相当 |
| 存储空间 | 大 | 小（几MB） |

## 注意事项

1. **GPU支持**: 代码自动检测CUDA，优先使用GPU
2. **模型加载**: 首次运行需要加载本地模型
3. **显存占用**: Qwen2.5-0.5B模型约需2-3GB显存
4. **批处理大小**: 默认batch_size=2，可根据显存调整
5. **梯度累积**: 默认accumulation_steps=4，等效batch_size=8

## 依赖环境

- Python 3.8+
- PyTorch >= 1.9.0
- Transformers >= 4.37.0
- PEFT >= 0.4.0
- Accelerate >= 0.20.0
- scikit-learn >= 0.24.0
- tqdm >= 4.62.0

## 快速开始

```bash
# 进入项目目录
cd llm_model

# 安装依赖
pip install -r requirements.txt

# 运行训练（默认10%数据+LoRA）
python train.py

# 查看结果
cat outputs/llm/training_report.txt
```

## 示例输出

使用采样功能后会显示：

```
加载模型: pretrain_models/Qwen2___5-0___5B-Instruct
LoRA配置: r=8, alpha=16, dropout=0.1
可训练参数: 1,234,567 / 615,234,567 (0.20%)

加载数据集...
随机采样数据集 (采样比例: 0.1)...
采样后训练集: 10,000条 (原始: 100,000条)
采样后验证集: 1,000条 (原始: 10,000条)
采样后测试集: 1,000条 (原始: 10,000条)

开始训练...
```

训练完成后会输出：

```
训练完成！
最优验证F1: 0.8523
测试集F1: 0.8456
所有结果保存在: outputs/llm/
```

## 预测示例

```
预测结果示例（前10条）:
--------------------------------------------------------------------------------
1. 句子1: 我想查询我的信用卡账单...
   句子2: 如何查看信用卡账单...
   生成: 是 | 预测: 1 | 实际: 1 | OK

2. 句子1: 信用卡怎么办理...
   句子2: 如何申请贷款...
   生成: 否 | 预测: 0 | 实际: 0 | OK
```

## 故障排除

### 1. 显存不足
- 减小batch_size（如改为1）
- 增加accumulation_steps（如改为8）
- 使用更小的max_len（如改为128）

### 2. 模型加载失败
- 检查模型路径是否正确
- 确保模型文件完整

### 3. LoRA保存失败
- 确保输出目录有写入权限
- 检查磁盘空间是否充足
