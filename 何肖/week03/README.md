# Week03 深度学习文本分类项目

## 项目概述
本项目实现了基于RNN和LSTM的中文文本多分类任务，用于预测句子中"你"字的位置。这是一个典型的序列标注问题转化为分类问题的案例。

## 项目结构
- `作业1.py` - 基于双向LSTM的文本分类模型
- `作业2.py` - 基于普通RNN的文本分类模型
- `update_2000_sentences.txt` - 训练数据文件
- `README.md` - 项目说明文档

## 任务描述
- **输入**：包含"你"字的中文句子（最长32个字符）
- **输出**："你"字在句子中的位置索引（0-31）
- **任务类型**：32分类问题

## 技术特点

### 作业1.py - 双向LSTM模型
- 使用双向LSTM捕捉前后文信息
- 字符级嵌入（Character-level Embedding）
- 最大池化（Max Pooling）提取关键特征
- 交叉熵损失函数（Cross Entropy Loss）
- 批归一化和Dropout防止过拟合

### 作业2.py - 普通RNN模型
- 使用普通RNN进行序列建模
- 字符级嵌入（Character-level Embedding）
- 最大池化（Max Pooling）提取关键特征
- 交叉熵损失函数（Cross Entropy Loss）
- Dropout防止过拟合

## 模型架构对比

### 作业1.py (双向LSTM)
### 作业2.py (普通RNN)

## 超参数配置
- 序列最大长度（MAXLEN）：32
- 词嵌入维度（EMBED_DIM）：64
- 隐藏层维度（HIDDEN_DIM）：128
- 学习率（LR）：0.001
- 批次大小（BATCH_SIZE）：64
- 训练轮数（EPOCHS）：40
- 训练/验证分割比例：80%/20%

## 数据处理
- 自动构建字符级词汇表
- 对句子进行编码和长度标准化
- 使用`<PAD>`和`<UNK>`特殊标记处理长度不一的句子

## 核心功能模块

### 1. 数据构建 (`build_dataset`)
- 从文本文件中读取包含"你"字的句子
- 提取"你"字的位置作为标签

### 2. 词汇表构建 (`build_vacab`)
- 构建字符级词汇表
- 包括特殊标记 `<PAD>` 和 `<UNK>`

### 3. 模型架构

#### 作业1.py (`multiClassRnn` with LSTM)
- 嵌入层：字符ID → 向量表示
- 双向LSTM：序列建模
- 最大池化：特征聚合
- 全连接层：分类输出

#### 作业2.py (`Model` with RNN)
- 嵌入层：字符ID → 向量表示
- 普通RNN：序列建模
- 最大池化：特征聚合
- 全连接层：分类输出

### 4. 训练和评估
- 训练循环：优化模型参数
- 评估函数：计算准确率

## 使用方法

### 1. 准备数据
将包含"你"字的中文句子保存到`update_2000_sentences.txt`文件中，每行一个句子。

### 2. 训练LSTM模型
```bash
python 作业1.py
```

### 3. 训练RNN模型
```bash
python 作业2.py
```

### 4. 模型评估
训练完成后，模型会自动在验证集上评估性能并输出准确率。

### 5. 推理示例
模型会自动对测试句子进行推理，输出预测的"你"字位置。

## 依赖库
- PyTorch
- NumPy
- JSON

## 性能指标
- 模型参数量
- 验证集准确率
- 训练/验证损失曲线

## 模型细节对比

### 作业1.py - 双向LSTM模型
- **嵌入层**: `nn.Embedding(vocab_size, embed_dim, padding_idx=0)`
- **LSTM层**: `nn.LSTM(embed_dim, hidden_dim, bias=True, batch_first=True, bidirectional=True)`
- **输出维度**: 因为是双向，输出维度为 `2 * hidden_dim`
- **特点**: 能够同时利用前向和后向的上下文信息

### 作业2.py - 普通RNN模型
- **嵌入层**: `nn.Embedding(vocab_size, embed_dim, padding_idx=0)`
- **RNN层**: `nn.RNN(embed_dim, hidden_dim, batch_first=True)`
- **输出维度**: 单向，输出维度为 `hidden_dim`
- **特点**: 简单的循环神经网络，只利用前向信息

## 输出处理
- 使用最大池化对序列信息进行聚合
- 通过全连接层将隐藏状态映射到32个类别
- 使用交叉熵损失函数进行训练

## 代码结构说明

### 共同模块
- `build_dataset()`: 构建训练数据集
- `build_vacab()`: 构建词汇表
- `encode()`: 文本编码函数
- `TextDataset`: 自定义数据集类
- `evaluate()`: 模型评估函数
- `train()`: 训练主函数

### 模型特定模块
- `作业1.py`: `Model` 使用双向LSTM
- `作业2.py`: `multiClassRnn` 使用普通RNN

## 注意事项
- 确保数据文件`update_2000_sentences.txt`位于同一目录
- 可根据需要调整超参数以获得更好的性能
- 模型支持中文字符的处理和分类
- 训练过程中会自动保存模型权重到`model.pth`
- 作业1.py使用双向LSTM，理论上能够更好地捕捉上下文信息
- 作业2.py使用普通RNN，结构更简单，但可能在长序列建模方面不如LSTM

## 模型对比
- **双向LSTM** (作业1.py): 更强的序列建模能力，能够利用前后文信息
- **普通RNN** (作业2.py): 结构简单，适合短序列任务，但可能存在梯度消失问题