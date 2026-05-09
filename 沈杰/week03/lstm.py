import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

# ===================== 固定你的 LSTM 核心实现 =====================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class DiyLSTM:
    def __init__(self, weight_ih, weight_hh, bias_ih, bias_hh, hidden_size):
        self.weight_ih = weight_ih
        self.weight_hh = weight_hh
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        h = np.zeros(self.hidden_size)
        c = np.zeros(self.hidden_size)
        output = []
        for xt in x:
            gates = (self.weight_ih @ xt + self.bias_ih
                     + self.weight_hh @ h + self.bias_hh)
            i = sigmoid(gates[:self.hidden_size])
            f = sigmoid(gates[self.hidden_size:2*self.hidden_size])
            g = tanh(gates[2*self.hidden_size:3*self.hidden_size])
            o = sigmoid(gates[3*self.hidden_size:])
            c = f * c + i * g
            h = o * tanh(c)
            output.append(h.copy())
        return np.array(output), h, c

# ===================== 任务：5字文本 + “你”字位置5分类 =====================
# 超参数
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

MAXLEN = 5
EMBED_DIM = 16
HIDDEN_DIM = 32
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
N_SAMPLES = 4000

# 随机汉字库
CHARS = ['我','他','她','它','好','坏','爱','恨','吃','喝','玩','乐','走','跑','看','听']

# ===================== 1. 数据集生成 =====================
def generate_5char_sent():
    """生成5字文本，恰好1个“你”，返回文本 + 位置类别(1~5)"""
    pos = random.randint(0,4)
    sent = []
    for i in range(5):
        sent.append('你' if i==pos else random.choice(CHARS))
    return ''.join(sent), pos+1

def build_dataset(n=N_SAMPLES):
    data = [generate_5char_sent() for _ in range(n)]
    random.shuffle(data)
    return data

# ===================== 2. 词表与编码 =====================
def build_vocab(data):
    vocab = {'<PAD>':0, '<UNK>':1}
    for s,_ in data:
        for c in s:
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab

def encode(s, vocab, maxlen=5):
    ids = [vocab.get(c,1) for c in s]
    return ids[:maxlen] + [0]*(maxlen-len(ids))

# ===================== 3. Dataset =====================
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(s,vocab) for s,_ in data]
        self.y = [lb-1 for _,lb in data]  # 0~4

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.tensor(self.x[i]), torch.tensor(self.y[i])

# ===================== 4. LSTM 分类模型（PyTorch官方LSTM） =====================
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, _) = self.lstm(x)
        feat = out.max(dim=1)[0]  # 序列池化
        logits = self.fc(feat)
        return logits

# ===================== 5. 训练与评估 =====================
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x,y in loader:
            pred = model(x).argmax(1)
            correct += (pred==y).sum().item()
            total += len(y)
    return correct/total

def train():
    # 数据
    data = build_dataset()
    vocab = build_vocab(data)
    split = int(0.8*len(data))
    train_loader = DataLoader(TextDataset(data[:split], vocab), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(data[split:], vocab), BATCH_SIZE)

    # 模型
    model = LSTMClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练
    print("开始训练 LSTM 5分类任务...\n")
    for ep in range(1, EPOCHS+1):
        model.train()
        loss_sum = 0
        for x,y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        acc = evaluate(model, val_loader)
        print(f"Epoch {ep:2d} | loss={loss_sum/len(train_loader):.4f} | val_acc={acc:.4f}")

    # 测试
    print("\n=== 测试结果：你在第几位 = 第几类 ===")
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
        for s in test_sents:
            idx = torch.tensor([encode(s,vocab)])
            p = model(idx).argmax().item()+1
            t = s.index('你')+1
            print(f"文本：{s} | 预测：第{p}类 | 真实：第{t}类")

if __name__ == '__main__':
    train()