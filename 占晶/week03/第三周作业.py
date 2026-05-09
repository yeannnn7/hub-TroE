"""
train_chinese_cls_rnn_homework.py
中文句子关键词分类 —— 简单 RNN 版本

任务：对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid
优化：Adam (lr=1e-3)   损失：交叉熵cross_entropy   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────

TEMPLATES_POS = [
    '天气晴朗',
    '比较平淡',
    '心情不错',
    '比较繁琐',
    '容易感冒',
    '耽误时间',
    '公交晚点',
    '做了很久',
    '排队结账',
    '路上堵车',
    '忘带雨伞',
]

#TEMPLATES_POS中随机选取一个句子，在随机一个位置插入你字，返回插入的位置
# def get_random_data():
#     # 随机选择一个句子
#     sentence = random.choice(TEMPLATES_POS)

#     # 随机选择一个插入位置（0 到 len(sentence) 之间，包括末尾）
#     insert_pos = random.randint(0, len(sentence))
#     new_sentence = sentence[:insert_pos] + '你' + sentence[insert_pos:]
#     return new_sentence, insert_pos

CHARS = list("我他天地人心情好坏高低美丑胖瘦早上中午晚上白天")

def get_random_data():
    sent = [random.choice(CHARS) for _ in range(5)]
    pos = random.randint(0, 4)
    sent[pos] = '你'
    return ''.join(sent), pos

def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n):
        result_sentence, position = get_random_data()
        data.append((result_sentence, position))
    random.shuffle(data)
    # with open('C:\\data.txt', 'w', encoding='utf-8') as f:
    #     for sentence, pos in data:
    #     # 格式：句子|位置
    #         f.write(f"{sentence}|{pos}\n")
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
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（RNN + MaxPooling 版）
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (MSELoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        #pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling
        #pooled = self.dropout(self.bn(pooled))
        #out = torch.sigmoid(self.fc(pooled).squeeze(1))  # (B,)
        last = e[:, -1, :]
        out = self.fc(last)
        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            digits = model(X)
            pred = torch.argmax(digits, dim=1)
            correct += (pred == y.long()).sum().item()
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

    model     = KeywordRNN(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
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
    #我他天地人心情好坏高低美丑胖瘦早上中午晚上白天
    test_sents = [
        '你白天晚上',
        '天高瘦午你',
        '坏你我他丑',
        '人天你胖人',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)                    # (1, 5)
            pred_class = torch.argmax(logits, dim=1).item()
            print(f"  [你在第 {pred_class + 1} 位]  {sent}")


if __name__ == '__main__':
    train()
