"""
train_position_cls.py
中文"你"字位置分类任务 —— RNN/LSTM 多分类版本

任务：对任意包含"你"字的5字文本，预测"你"在第几位（1-5类）
模型：Embedding → RNN/LSTM → 取最后隐藏状态 → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   CPU运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 5000  # 样本总数
MAXLEN = 5  # 固定5个字
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 15
TRAIN_RATIO = 0.8
USE_LSTM = True  # True=LSTM, False=RNN

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用汉字（排除"你"）
COMMON_CHARS = [
    '我', '他', '她', '它', '们', '的', '了', '在', '是', '有',
    '不', '人', '都', '一', '个', '大', '上', '年', '这', '中',
    '说', '去', '好', '小', '会', '来', '后', '着', '和', '就',
    '到', '过', '可', '出', '能', '没', '对', '起', '也', '天',
    '得', '那', '要', '下', '里', '看', '时', '生', '自', '方',
    '而', '前', '开', '心', '些', '现', '又', '很', '其', '明',
    '知', '问', '法', '点', '意', '事', '两', '次', '使', '身',
    '被', '从', '已', '子', '工', '也', '如', '经', '头', '面',
]

YOU_CHAR = '你'


def generate_sample():
    """生成一个样本：随机决定"你"的位置，填充其他字符"""
    position = random.randint(0, 4)  # 0-4对应第1-5位

    # 生成5个字符，在指定位置插入"你"
    chars = []
    for i in range(5):
        if i == position:
            chars.append(YOU_CHAR)
        else:
            chars.append(random.choice(COMMON_CHARS))

    sentence = ''.join(chars)
    label = position  # 标签为0-4
    return sentence, label


def build_dataset(n=N_SAMPLES):
    """构建数据集"""
    data = [generate_sample() for _ in range(n)]
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    """构建字符级词表"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    """将句子编码为索引序列"""
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))  # padding
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 多分类用long类型
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionClassifier(nn.Module):
    """
    "你"字位置分类器
    架构：Embedding → RNN/LSTM → 取最后隐藏状态 → Linear → Softmax
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=5, use_lstm=USE_LSTM):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 可选择RNN或LSTM
        if use_lstm:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (B, L, embed_dim)
        output, hidden = self.rnn(embedded)  # (B, L, hidden_dim)

        # 取最后一个时间步的隐藏状态
        if isinstance(hidden, tuple):  # LSTM返回(h_n, c_n)
            last_hidden = hidden[0][-1]  # 取h_n的最后一层
        else:  # RNN直接返回h_n
            last_hidden = hidden[-1]  # 取最后一层

        logits = self.fc(last_hidden)  # (B, num_classes)
        return logits


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    """评估准确率"""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)  # 取概率最大的类别
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    # 打印几个样本示例
    print("\n  样本示例：")
    for i in range(5):
        sent, label = data[i]
        print(f"    '{sent}' → 第{label + 1}位")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(PositionDataset(train_data, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PositionDataset(val_data, vocab),
                            batch_size=BATCH_SIZE)

    model_name = "LSTM" if USE_LSTM else "RNN"
    print(f"\n使用模型：{model_name}")
    model = PositionClassifier(vocab_size=len(vocab), use_lstm=USE_LSTM)

    criterion = nn.CrossEntropyLoss()  # 多分类用交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    best_val_acc = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最佳验证准确率：{best_val_acc:.4f}")

    # 推理示例
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你今天好吗',  # 第1位
        '你好世界呀',  # 第1位
        '今天你在哪',  # 第3位
        '明天见你哦',  # 第4位
        '早上好你呀',  # 第5位
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
            confidence = prob[0][pred].item()
            print(f"  '{sent}' → 第{pred + 1}位 (置信度: {confidence:.2%})")


if __name__ == '__main__':
    train()
