"""
train_chinese_cls_rnn.py
5个字文本中"你"字位置分类 —— 简单 RNN 版本

任务：对5个字的文本，"你"在第几位就属于第几类（1-5）
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Softmax
优化：Adam (lr=1e-3)   损失：CrossEntropyLoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 2000
MAXLEN      = 5
EMBED_DIM   = 64

HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLASSES = 5  # "你"字可能出现的位置：第1-5位

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 生成5个字的文本，"你"在第几位就属于第几类

CHARS = '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产样配到却多打一个人人了已百个人'

def make_sample():
    """生成一个样本：5个字的文本，"你"在某一位"""
    pos = random.randint(0, NUM_CLASSES - 1)  # 0-4 对应第1-5位
    
    # 生成其他4个字
    other_chars = [random.choice(CHARS) for _ in range(4)]
    
    # 构建文本：在pos位置插入"你"
    text = ''.join(other_chars[:pos]) + '你' + ''.join(other_chars[pos:])
    
    # 标签：0-4 分别对应"你"在第1-5位
    label = pos
    
    return text, label


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        data.append(make_sample())
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 改为long类型（CrossEntropyLoss需要）
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class CharPositionRNN(nn.Module):
    """
    5字文本中"你"字位置分类器（5分类 RNN 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        pooled = self.dropout(self.bn(pooled))
        out = self.fc(pooled)               # (B, num_classes)
        return out  # 返回logits，由CrossEntropyLoss处理


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits  = model(X)
            pred    = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total   += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = CharPositionRNN(vocab_size=len(vocab), num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    # 生成几个包含"你"字的5字文本
    test_sents = [
        '你好美丽啊',
        '他是你的人',
        '我非常爱你',
        '她爱你万年',
        '我不需要你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids    = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred   = torch.argmax(logits, dim=1).item()
            probs  = torch.softmax(logits, dim=1)[0]
            pos    = pred + 1  # 转换为第1-5位
            print(f"  [第{pos}位]  {sent}  置信度{probs[pred]:.4f}")


if __name__ == '__main__':
    train()
