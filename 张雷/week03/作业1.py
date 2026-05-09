"""
设计一个以文本为输入的多分类任务实验一下用RNN,LSTM等模型的跑通训练。
任务: 对一个任意包含"你"字的五个字的文本，"你"在第几位，就属于第几类。
对比模型: RNN, LSTM, GRU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt

# -------------------- 1. 生成数据集 --------------------
CHAR_POOL = list("我你是他她它很好吗的了一不人大小中上下来去天工和机学会可要产民对能行方说这时那也里后前在有个到出过得子开着道看场面把样关点心然现想起经发理用家意成所事法没如还问话知信重体相东路已手都题自量明实物从当气本打做此进力内平实加回定总数正比老很名高文公战国水青红头问记组特表神教太眼长声府区快技济族早马夜嗯哦呵嘛嘿哎喂")

def generate_sample():
    pos = random.randint(0, 4)
    chars = random.choices(CHAR_POOL, k=5)
    chars[pos] = "你"
    return "".join(chars), pos

def generate_dataset(n):
    samples = [generate_sample() for _ in range(n)]
    texts, labels = zip(*samples)
    return list(texts), list(labels)

# -------------------- 2. 构建词表 --------------------
def build_vocab(chars):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab
VOCAB = build_vocab(sorted(set("".join(CHAR_POOL))))

print(f"词表: {VOCAB}")
print(VOCAB.get("你"))
VOCAB_SIZE = len(VOCAB)
NUM_CLASSES = 5

def text_to_tensor(text):
    return torch.tensor([VOCAB.get(ch, 1) for ch in text], dtype=torch.long)

# -------------------- 3. Dataset --------------------
class CharPositionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            text_to_tensor(self.texts[idx]), 
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -------------------- 4. 模型定义 --------------------
class KeywordClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, rnn_type="rnn",dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_type = rnn_type
        if rnn_type == "lstm":
            self.model = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        elif rnn_type == "gru":
            self.model = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        else:
            self.model = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        if self.rnn_type == "lstm":
            e, (h_n, _) = self.model(x)
        else:
            e, h_n = self.model(x)
        pooled = e.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return self.fc(pooled)

# -------------------- 5. 训练与评估 --------------------
def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        optimizer.zero_grad() # 反向传播前先清零梯度
        logits = model(x) # 前向传播得到预测结果
        loss = loss_fn(logits, y) # 计算损失
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 更新模型参数
        total_loss += loss.item() * x.size(0) # 累积损失
        correct += (logits.argmax(1) == y).sum().item() # 累积正确预测数
        total += x.size(0) # 累积样本总数
    return total_loss / total, correct / total

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x) # 前向传播得到预测结果
            loss = loss_fn(logits, y) # 计算损失
            total_loss += loss.item() * x.size(0) # 累积损失
            correct += (logits.argmax(1) == y).sum().item() # 累积正确预测数
            total += x.size(0) # 累积样本总数
    return total_loss / total, correct / total

def train_model(model, train_loader, test_loader, epochs, lr, name):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, loss_fn, optimizer)
        te_loss, te_acc = evaluate(model, test_loader, loss_fn)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        if epoch % 5 == 0:
            print(f"[{name}] Epoch {epoch:3d}: train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f}, "
                  f"test_loss={te_loss:.4f}, test_acc={te_acc:.4f}")
    return history

# -------------------- 6. 主流程 --------------------
def main():
    random.seed(42)
    torch.manual_seed(42)

    train_texts, train_labels = generate_dataset(8000)
    test_texts, test_labels = generate_dataset(2000)

    train_dataset = CharPositionDataset(train_texts, train_labels)
    test_dataset = CharPositionDataset(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\n样例数据:")
    for i in range(6):
        print(f"  文本: '{train_texts[i]}' -> 类别: {train_labels[i]}")

    EMBED_DIM = 32
    HIDDEN_DIM = 64
    EPOCHS = 30
    LR = 0.001

    models = {
        "RNN":  KeywordClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, "rnn"),
        "LSTM": KeywordClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, "lstm"),
        "GRU":  KeywordClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, "gru"),
    }
    for model in models:
        print(f"\n--- 训练 {model} 模型 ---")
        train_model(models[model], train_loader, test_loader, EPOCHS, LR, model)
        _, test_acc = evaluate(models[model], test_loader, nn.CrossEntropyLoss())
        print(f"\n最终验证准确率：{test_acc:.4f}")

        print("\n--- 推理示例 ---")
        models[model].eval()
        test_samples = [
            ("你人大中小", 0),
            ("人你大中小", 1),
            ("人大你中小", 2),
            ("人大中你小", 3),
            ("你大中小你", 0),
        ]
        for text, true_label in test_samples:
            input_tensor = text_to_tensor(text).unsqueeze(0) # 转换为批次格式
            with torch.no_grad():
                pred = models[model](input_tensor).argmax(1).item()
            correct = "✓" if pred == true_label else "✗"
            print(f"  文本: '{text}' -> 预测: {pred}  真实: {true_label} {correct}")


if __name__ == "__main__":
    main()
