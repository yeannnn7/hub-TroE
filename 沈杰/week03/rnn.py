import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000    # 样本数量
MAXLEN      = 5       # 固定5个字
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 15
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成：生成5个字文本，包含一个“你”字 ─────────────────
# 随机汉字库（不含你）
CHARS = ['我', '他', '她', '它', '们', '好', '坏', '爱', '恨', '喜',
         '欢', '吃', '喝', '玩', '乐', '走', '跑', '看', '听', '说',
         '大', '小', '多', '少', '高', '低', '快', '慢', '来', '去']

def generate_5char_sentence():
    """生成固定5个字的句子，恰好包含1个“你”字，位置随机"""
    pos = random.randint(0, 4)  # 0~4 对应第1~5位
    sent = []
    for i in range(5):
        if i == pos:
            sent.append('你')
        else:
            sent.append(random.choice(CHARS))
    return ''.join(sent), pos + 1  # 返回句子 + 类别（1-5类）

def build_dataset(n=N_SAMPLES):
    """生成数据集：5字文本 + 你字位置类别"""
    data = []
    for _ in range(n):
        sent, label = generate_5char_sentence()
        data.append((sent, label))
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
        self.y = [lb-1 for _, lb in data]  # 标签转0~4（适配交叉熵）

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 分类标签
        )

# ─── 4. 模型定义：5分类任务 ─────────────────────────────────
class PositionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, num_classes)  # 输出5类

    def forward(self, x):
        # x: (batch, 5)
        e, _ = self.rnn(self.embedding(x))  # (B,5,H)
        feat = e.max(dim=1)[0]              # (B,H)
        feat = self.dropout(feat)
        logits = self.fc(feat)              # (B,5)
        return logits

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total

def train():
    print("生成5字文本数据集（含“你”字）...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = PositionClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()  # 多分类损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("开始训练...\n")
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
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    # ─── 测试推理 ─────────────────────────────────────
    print("\n=== 测试结果（你在第几位=第几类）===")
    test_sents = [
        '你一一一一',  
        '一你一一一',   
        '一一你一一',  
        '一一一你一',  
        '一一一一你',   
        '你一一一一',   
        '一你一一一',  
    ]
    
    model.eval()
    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_cls = torch.argmax(logits).item() + 1  # 转回1-5
            true_pos = sent.index('你') + 1
            print(f"文本：{sent} | 预测第{pred_cls}类 | 真实第{true_pos}类")

if __name__ == '__main__':
    train()