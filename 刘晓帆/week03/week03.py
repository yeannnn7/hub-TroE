import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ==================== 1. 数据集构建 ====================
# 字符映射表（只使用汉字部分）
char_pool = list("天地人你我看他她它这那大小多少上下左右前后里外东西南北春夏秋冬风雨云山花鸟树石水火土日月星心口手足耳目语言思量行走坐卧吃喝玩乐")

# 构建词汇表（添加<pad>和<unk>）
vocab = {ch: idx for idx, ch in enumerate(char_pool, start=2)}  # 索引从2开始
vocab["<pad>"] = 0
vocab["<unk>"] = 1
vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")

def generate_sample():
    """生成一个5字文本，'你'在随机位置，返回文本和标签"""
    pos = random.randint(0, 4)  # 标签：0-4
    chars = []
    for i in range(5):
        if i == pos:
            chars.append("你")
        else:
            c = "你"
            while c == "你":  # 确保其他位置不是"你"
                c = random.choice(char_pool[1:])  # 跳过char_pool中的第一个"你"
            chars.append(c)
    text = "".join(chars)
    return text, pos

class TextDataset(Dataset):
    def __init__(self, num_samples, vocab):
        self.samples = []
        self.labels = []
        self.vocab = vocab
        
        for _ in range(num_samples):
            text, label = generate_sample()
            # 将文本转为索引序列
            indices = [vocab.get(ch, vocab["<unk>"]) for ch in text]
            self.samples.append(torch.tensor(indices))
            self.labels.append(torch.tensor(label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# 生成数据
train_dataset = TextDataset(5000, vocab)
test_dataset = TextDataset(1000, vocab)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# 验证数据分布
labels = [label.item() for _, label in train_dataset]
print(f"训练集类别分布: {np.bincount(labels)}")

# ==================== 2. 模型定义 ====================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        out, _ = self.rnn(x)           # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]            # 取最后一个时间步的输出
        return self.fc(out)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_dim)
        out = out[:, -1, :]            # 取最后时刻的输出
        return self.fc(out)

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==================== 3. 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    return total_loss / total, correct / total

# ==================== 4. 训练流程 ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备: {device}")

# 超参数
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5
EPOCHS = 20
LR = 0.001

# 初始化三个模型
models = {
    "RNN": RNNClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device),
    "LSTM": LSTMClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device),
    "GRU": GRUClassifier(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
}

criterion = nn.CrossEntropyLoss()
histories = {}

print("\\n" + "="*50)
print("开始训练...")
print("="*50)

for model_name, model in models.items():
    print(f"\\n--- 训练 {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    histories[model_name] = history

# ==================== 5. 对比可视化 ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
for name, hist in histories.items():
    axes[0].plot(range(1, EPOCHS+1), hist["test_loss"], label=name, linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Test Loss Comparison")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 准确率曲线
for name, hist in histories.items():
    axes[1].plot(range(1, EPOCHS+1), hist["test_acc"], label=name, linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Test Accuracy Comparison")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("rnn_comparison.png", dpi=150)
plt.show()

# ==================== 6. 最终报告 ====================
print("\\n" + "="*50)
print("最终结果对比:")
print("="*50)
for name, hist in histories.items():
    final_acc = hist["test_acc"][-1]
    print(f"{name:4s}: {final_acc:.4f}")

# ==================== 7. 交互测试 ====================
print("\\n" + "="*50)
print("交互测试样例:")
print("="*50)

best_model = models["GRU"]  # GRU 通常收敛较快
best_model.eval()

test_texts = [
    "你我他天地",  # 你第0位 → 类别0
    "是我你看书",  # 你第2位 → 类别2
    "天地人你我",  # 你第3位 → 类别3
    "看你去哪里",  # 你第1位 → 类别1
    "他她你它我",  # 你第2位 → 类别2
]

with torch.no_grad():
    for text in test_texts:
        indices = [vocab.get(ch, vocab["<unk>"]) for ch in text]
        x = torch.tensor([indices]).to(device)
        output = best_model(x)
        print(f"'output:' {output}")

        pred = torch.argmax(output, dim=1).item()
        true_pos = text.index("你")
        mark = "✓" if pred == true_pos else "✗"
        print(f"'{text}' → 真实位置: {true_pos}, 预测: {pred} {mark}")
