import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#- 超参数
N_SAMPLES = 2000
EMBED_DIM = 64
HIDDEN_DIM = 64
TRAIN_RATIO = 0.8
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 20

# -数据生成相关的代码-
# 固定字表（避免生成奇怪字符）
CHAR_POOL = list("今天天气很好我们一起去学习吃饭玩耍快乐生活真的非常不错啊吧呢吗啦哈嘿呀")

TARGET_CHAR = "你"
SEQ_LEN = 5

def generate_sample():
    """
    生成一个样本：
    - 长度固定为5
    - 必须包含“你”
    - 返回：(sentence, label)
    label = “你”的位置（0~4）
    """
    # 随机选择“你”的位置
    pos = random.randint(0, SEQ_LEN - 1)

    chars = []
    for i in range(SEQ_LEN):
        if i == pos:
            chars.append(TARGET_CHAR)
        else:
            chars.append(random.choice(CHAR_POOL))

    sentence = "".join(chars)

    return sentence, pos

def build_dataset(N_SAMPLES):
    data = []
    for _ in range(N_SAMPLES):
        data.append(generate_sample())

    random.shuffle(data)
    return data

# 词表构建与编码
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab    

def encode(sent, vocab, maxlen=SEQ_LEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

# Dataset / DataLoader
class MyDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [label for _, label in data]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return(
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)
        )

# 模型定义
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim = EMBED_DIM, hidden_dim = HIDDEN_DIM, dropout = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, SEQ_LEN)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        pooled = x.max(dim=1).values
        pooled = self.dropout(pooled)
        out = self.fc(pooled)
        return out


# 评估函数
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            pred_labels = pred.argmax(dim = 1)
            correct += (pred_labels == y).sum().item()
            total += y.size(0)
    return correct / total

# 保存模型
def save_model(model, vocab, path="model.pth"):
    """
    保存模型参数 + vocab
    """
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab
    }, path)

    print(f"✅ 模型已保存到: {path}")

# 训练
def train():
    # 1. 数据准备
    data = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f" 样本数:{len(data)} 词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]
    
    train_loader = DataLoader(MyDataset(train_data, vocab), batch_size = BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MyDataset(val_data, vocab), batch_size = BATCH_SIZE)
    
    # 2. 模型、损失函数、优化器
    model = KeywordRNN(len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS} loss = {avg_loss:.4f} val_acc = {val_acc:.4f}")
    
    final_acc = evaluate(model, val_loader)
    print(f"\n最终验证准确率：{final_acc:.4f}")
    save_model(model, vocab)

    
# 加载模型
def load_model(path="model.pth"):
    checkpoint = torch.load(path)

    vocab = checkpoint["vocab"]
    model = KeywordRNN(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab
# 推理函数
def predict(model, vocab, sentence):
    ids = encode(sentence, vocab)
    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return pred

# 推理示例
def demo_inference():
    print("\n--- 加载模型后的推理示例 ---")

    # 加载模型
    model, vocab = load_model()

    # 手动测试句子
    test_sents = [
        "你今天很好",
        "天气你很好",
        "今天很好你",
        "很好你今天",
        "今你天很好"
    ]

    for sent in test_sents:
        pred = predict(model, vocab, sent)
        true_pos = sent.index("你")
        print(f"句子: {sent}  | 真实: {true_pos}  | 预测: {pred}")


if __name__ == "__main__":
    mode = input("train / demo: ")

    if mode == "train":
        train()
    elif mode == "demo":
        demo_inference()

