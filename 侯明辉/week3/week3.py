"""
任务定义：
  - 输入：最长 32 字的中文句子，其中包含一个特定字符 '*'
  - 输出：'*' 首次出现的位置索引

数据生成：
  - 从中文句子模板池中随机选取句子
  - 在句子中随机位置插入 '*'
  - 30% 概率再插入第二个 '*' 作为干扰（标签仍是首次出现位置）

模型：RNN / LSTM / Linear 
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
#SEED        = 42
N_SAMPLES   = 6000
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 10
TRAIN_RATIO = 0.8
SPECIAL_CHAR = '*'  # 特定字符

#random.seed(SEED)
#torch.manual_seed(SEED)

# ─── 1. 数据生成（中文句子 + 随机插入特定字符）────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SENTENCES_FILE = os.path.join(SCRIPT_DIR, 'sentences.txt')


def load_sentences(filepath=SENTENCES_FILE):
    """从 txt 文件读取中文句子，每行一条，自动去除空行和首尾空白"""
    with open(filepath, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    print(f"  从 {os.path.basename(filepath)} 加载了 {len(sentences)} 条句子")
    return sentences


def make_sample(sentences):
    base = random.choice(sentences)

    insert_pos = random.randint(0, len(base))
    text = base[:insert_pos] + SPECIAL_CHAR + base[insert_pos:]

    text = text[:MAXLEN]
    # * 首次出现的位置
    star_pos = text.index(SPECIAL_CHAR)
    # 以一定概率在后面再插入一个 *（增加干扰，标签仍是首次出现位置）
    if random.random() < 0.3 and len(text) < MAXLEN:
        second_pos = random.randint(star_pos + 1, len(text))
        text = text[:second_pos] + SPECIAL_CHAR + text[second_pos:]
        text = text[:MAXLEN]
    return text, star_pos


def build_dataset(sentences, n=N_SAMPLES):
    data = []
    for _ in range(n):
        text, pos = make_sample(sentences)
        data.append((text, pos))
    random.shuffle(data)
    return data

# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for text, _ in data:
        for ch in text:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

def encode(text, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in text]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# ─── 3. Dataset / DataLoader ────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [pos for _, pos in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 分类标签用 long
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class PositionRNN(nn.Module):
    """RNN 版本：Embedding → RNN → MaxPool → BN → Dropout → Linear"""
    def __init__(self, vocab_size, num_classes=MAXLEN,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))    # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]               # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)               # (B, num_classes)
        return logits


class PositionLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes=MAXLEN,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        e, _ = self.lstm(self.embedding(x))    # (B, L, hidden_dim)
        pooled = e.max(dim=1)[0]               # (B, hidden_dim)
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)               # (B, num_classes)
        return logits


class PositionLinear(nn.Module):
    def __init__(self, vocab_size, num_classes=MAXLEN,
                 embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.bn        = nn.BatchNorm1d(embed_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        e = self.embedding(x)                   
        pooled = e.max(dim=1)[0]                
        pooled = self.dropout(self.bn(pooled))
        logits = self.fc(pooled)                
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train_model(model, train_loader, val_loader, name, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"模型: {name}  参数量: {total_params:,}")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"最终验证准确率: {final_acc:.4f}")
    return model, final_acc

# ─── 6. 推理演示 ────────────────────────────────────────────
def demo_inference(model, vocab, name):
    model.eval()
    test_cases = [
        '今天天*气不错',
        '*这部电影很好看',
        '明天要去开*会',
        '晚饭想*吃火锅',
        '*最近在学习编程',
        '**周末去公园散步',    
        '今天工作很顺利**',  
    ]
    print(f"\n--- {name} 推理示例 ---")
    with torch.no_grad():
        for text in test_cases:
            ids   = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(ids)
            pred   = logits.argmax(dim=1).item()
            prob   = torch.softmax(logits, dim=1).max().item()

            true_pos = text.index(SPECIAL_CHAR) if SPECIAL_CHAR in text else -1
            mark = "OK" if pred == true_pos else "POK"
            print(f"  {mark} 输入: {text!r:30s}  预测位置: {pred:2d}  真实位置: {true_pos:2d} ")

# ─── 7. 主入口 ──────────────────────────────────────────────
def main():
    print("加载中文句子...")
    sentences = load_sentences()

    print("生成数据集...")
    data  = build_dataset(sentences, N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数: {len(data)}，词表大小: {len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(PositionDataset(train_data, vocab),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(PositionDataset(val_data, vocab),
                              batch_size=BATCH_SIZE)

    models = {
        'RNN':    PositionRNN(len(vocab)),
        'LSTM':   PositionLSTM(len(vocab)),
        'Linear': PositionLinear(len(vocab)),
    }

    results = {}
    for name, model in models.items():
        trained, acc = train_model(model, train_loader, val_loader, name)
        results[name] = (trained, acc)
        demo_inference(trained, vocab, name)

    print(f"\n{'='*60}")
    print("模型对比汇总")
    print(f"{'='*60}")
    for name, (_, acc) in sorted(results.items(), key=lambda x: -x[1][1]):
        print(f"  {name:8s}  验证准确率: {acc:.4f}")

if __name__ == '__main__':
    main()
