"""
文本分类任务：
    句子长度5个字，“你”在句中第几个位置，就是第几个类别
    1. 数据生成
    2. 词表构建与编码
    3. 模型构建
    4. 训练
    5. 评估
    6. 预测

"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 3000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 0x4e00 ~ 0x9fff 是 CJK 统一汉字基本区的 Unicode 编码范围
# 随机生成不包含“你”字的汉字
def random_chinese_char(exclu="你"):
    char = chr(random.randint(0x4e00, 0x9fff))
    while char == exclu:
        chr(random.randint(0x4e00, 0x9fff))
    return char

# 随机生成n条中文句子，每个句子长度为sl，有且只能有一个“你”字
def generate_chinese_sentences(n, sl=5):
    sentences = []
    sent = ""
    for _ in range(n):
        for _ in range(sl - 1):
            sent += random_chinese_char()
        pos = random.randint(0, sl - 1)
        sent = sent[:pos] + "你" + sent[pos:]
        sentences.append((sent, pos))
        sent = ""

    return sentences

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for st, _ in data:
        for ch in st:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(sent, vocab, maxlen=5):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]     # index of "你"

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
    架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Sigmoid → (CrossEntropyLoss)
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
        e, _ = self.rnn(self.embedding(x))  # (B, seq_len, hidden_dim)
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))  # (B, hidden_dim)
        # out = torch.sigmoid(self.fc(pooled))  # (B, 5)
        out = self.fc(pooled)  # (B, 5)
        return out
    
# ─── 5. 训练与评估 ──────────────────────────────────────────
# 测试每轮的准确率
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X)             # (B, 5)
            pred = pred.argmax(dim=1)   # (B,)
            # 比较pred 和y相同元素个数
            correct += (pred == y).sum().item()
            total += len(y)

    return correct / total


def train():
    print("生成数据集...")
    data  = generate_chinese_sentences(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    # print(vocab)
    # input()

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
            pred = model(X)             # (B, 5)
            loss = criterion(pred, y)   # (B, 5)
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
    test_sents = [
        '你好哈哈哈',
        '真棒你服务',
        '打你不停下',
        '公交不等你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            pred  = model(ids)
            pred  = pred.argmax(dim=1).item()
            print(f"测试句子：{sent}, 标签：{pred}")


if __name__ == '__main__':
    train()
