# 对一个任意包含你的五个字的文本，在第几位，就属于第几类
# 例如：'我是一个学生'，在第1位是'我'，所以属于第1类    
# 模型：Embedding + LSTM + MaxPool1d + Dropout + FC + sigmoid
# loss: cross_entropy
# 优化器：Adam
# 学习率：0.001
# 训练轮数：20
# 验证集准确率：0.95
# 测试集准确率：0.95
# 模型保存：model.pth
# 模型加载：model.pth
# 模型评估：evaluate(model,val_loader)

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
N_SAMPLES = 4000
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT = 0.3
TRAIN_RATIO = 0.7
EMBED_DIM = 64
HIDDEN_SIZE = 64
# 1.数据生成
word_keys= ['我','是','一','学','生','好', '棒', '赞', '喜','欢', '满','意','这','家','真','的','很','下','次','还','来']
def build_list():
    list = random.sample(word_keys, 4)
    num = random.randint(0,5)
    list.insert(num, '你')
    return list, num+1

def build_dataset():
    dataset = []
    for i in range(N_SAMPLES):
        list, num = build_list()
        dataset.append((list, num))
    random.shuffle(dataset)
    return dataset

# 2.词表构建与编码
def encode(sent,vocab):
    ids = [vocab.get(ch,vocab['<UNK>']) for ch in sent]
    return ids


def build_vocab():
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for index,item in enumerate(word_keys):
        vocab[item] = index + 2
    return vocab


class MyDataset(Dataset):
    def __init__(self, dataset,vocab):
        self.X = [encode(item[0],vocab) for item in dataset]
        self.y = [item[1] for item in dataset]
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long), 
            torch.tensor(self.y[idx], dtype=torch.float)
                )

# 3.模型定义
class LSTMModel(nn.Module):
    def __init__(self,vocab,embed_dim = EMBED_DIM,hidden_size = HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), embed_dim,padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_size, batch_first=True)
        self.maxpool = nn.MaxPool1d(1)
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self,x):
        x = self.embedding(x) # (batch_size, seq_len, embed_dim)
        x, _ = self.lstm(x) # (batch_size, seq_len, hidden_size)
        # x = self.maxpool(x)
        x = x.max(dim=1)[0] # (batch_size, hidden_size)
        x = self.batchnorm(x) # (batch_size, hidden_size)
        x = self.dropout(x) # (batch_size, hidden_size)
        x = self.fc(x) # (batch_size, 1)
        x = x.squeeze(1) # (batch_size)
        x = torch.sigmoid(x) # (batch_size)
        return x

# 模型验证
def evaluate(model,loader):
    model.eval()
    corrent = total = 0
    with torch.no_grad():
        for X,y in loader:
            pred = model(X)
            print('pred',pred)
            corrent += (pred == y).sum().item()
            total += len(y)
    print(f'corrent：{corrent}，total：{total}，acc：{corrent/total:.4f}')
    return corrent / total


def train():
    dataset = build_dataset()
    vocab = build_vocab()
    split = int(len(dataset) * TRAIN_RATIO)
    train_data = dataset[:split]
    val_data   = dataset[split:]
    traindata_loader = DataLoader(MyDataset(train_data,vocab), batch_size=BATCH_SIZE, shuffle=True)
    valdata_loader   = DataLoader(MyDataset(val_data,vocab), batch_size=BATCH_SIZE)
    model = LSTMModel(vocab)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数数量：{total_params}')
    for epoch in range(1):
    # for epoch in range(EPOCHS):
        # 模型训练
        model.train()
        total_loss = 0.0
        for X, y in traindata_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(traindata_loader)
        val_acc = evaluate(model,valdata_loader)
        print(f'轮次：{epoch+1}，avg_loss：{avg_loss:.4f}，val_acc：{val_acc:.4f}')
    
    print('模型训练完成,开始测试')
    with torch.no_grad():
        model.eval()
        # 1 2 5 3 3 3 4
        test_str = [
            ['你','是','一','学','生'],
            ['好','你','赞','喜','欢'],
            ['满','意','这','家','你'],
            ['真','的','你','下','次'],
            ['还','来','你','很','下'],
            ['很','下','次','还','来'],
            ['还','来','你','很','下'],
            ['很','下','次','你','来'],
        ]
        for str in test_str:
            X = torch.tensor([encode(str,vocab)], dtype=torch.long)
            pred = model(X).item()
            print(pred)

train()
