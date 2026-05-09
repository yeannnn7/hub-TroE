import os
import torch
import torch.nn as nn
from torch.optim import adam
import CreateSample as cs
from torch.utils.data import Dataset, DataLoader
import os


class RnnLstmModel(nn.Module):
    '''
    RNN与LSTM模型训练一个多分类任务，识别出五个字里“你”所在的位置作为类别
    下面是设计到的所有的组件
    Adam
    Embedding
    CNN
    RNN/LSTM
    Pooling
    Norm
    Dropout
    '''

    #-----超参数-
    TRAIN_SAMPLES_NUM = 4000 #定义训练样本的数量
    LR = 1e-3 #学习率
    SEN_MAX_LEN = 5 #句子的最大长度
    EMBED_DIM = 6 #词嵌入维度
    HIDDEN_DIM = 12 #RNN隐藏层的维度
    BATCH_SIZE = 20 #批次大小
    EPOCHS = 10 #训练轮数
    TRAIN_RATIO = 0.8 #百分之八十样本用来训练
    #-----------

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        #词嵌入
        self.embedding = nn.Embedding(vocab_size, embed_dim,padding_idx=0)
        #rnn层 rnn的输入形状 输入层、隐藏层
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        #线性层
        self.linear = nn.Linear(hidden_dim, 5)
    
    def forward(self,x):
        '''
        前向传播
        '''
        x_emb = self.embedding(x)
        # RNN返回两个值: output(所有时刻隐藏状态), hidden(最后时刻隐藏状态)
        o, h = self.rnn(x_emb) 
        # RNN输出维度是 (batch_size, seq_len, hidden_dim)
        # 因为这是分类任务，我们只关心最后得到的结果，取所有时刻的最大池化或最后一个时刻
        pooled = o.max(dim=1)[0]  # (batch_size, hidden_dim)
        #归一化层
        pooled = self.bn(pooled)
        #dropout层
        pooled = self.dropout(pooled)
        # 经过线性层映射到5个类别
        y_p = self.linear(pooled)
        return y_p

#构建词表
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence,_ in data:
        for char in sentence:
            if char not in vocab:
                vocab[char] = len(vocab)
    #print(vocab)
    return vocab


#数据集切片
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )



def encode(sent, vocab, maxlen=RnnLstmModel.SEN_MAX_LEN):
    ids = []
    for ch in sent:
        ids.append(vocab.get(ch, 1))
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            y = y.long()
            pred = model(x)
            # 因为是多分类(0-4)，用argmax找最大概率的索引作为预测类别
            pred_class = torch.argmax(pred, dim=-1)
            correct += (pred_class == y).sum().item()
            total += len(y)
    return correct / total

#开始训练模型
def train():
    '''
    主函数,训练并看效果
    '''
    #生成训练的数据集
    train_data = cs.create_samples(RnnLstmModel.TRAIN_SAMPLES_NUM)
    # print(train_data)
    print(f"训练数据集已生辰，大小为：{len(train_data)}")
    print(f"开始构建词表... ...")
    vocab = build_vocab(train_data)
    #print(vocab)
    print(f"词表已构建，大小为：{len(vocab)}")

    #切分训练与测试数据集
    split      = int(len(train_data) * RnnLstmModel.TRAIN_RATIO)
    #这是训练集
    train_data_split = train_data[:split]
    #这是验证集
    val_data   = train_data[split:]

    #数据集切片
    train_loder = DataLoader(TextDataset(train_data_split, vocab), batch_size=RnnLstmModel.BATCH_SIZE, shuffle=True)
    val_loader  = DataLoader(TextDataset(val_data,   vocab), batch_size=RnnLstmModel.BATCH_SIZE)
    print(train_loder)
    #实例化模型
    model = RnnLstmModel(vocab_size=len(vocab))
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #选择优化器
    adam = torch.optim.Adam(model.parameters(), lr=RnnLstmModel.LR)
    #开始训练
    for epoch in range(1, RnnLstmModel.EPOCHS + 1):
        #训练标志
        model.train()
        #损失值
        total_loss = 0.0
        for x, y in train_loder:
            # y是Long类型（交叉熵要求分类标签为LongTensor）
            y = y.long()
            
            #预测值
            pred = model(x)
            #当前损失值
            loss = criterion(pred, y)
            #梯度清零
            adam.zero_grad()
            #反向传播更新
            loss.backward()
            adam.step()
            total_loss += loss.item()
        #本轮岁损失值平均数
        avg_loss = total_loss / len(train_loder)
        print(f"本轮训练损失值{avg_loss}")
        #验证集准确率
        val_acc = evaluate(model, val_loader)
        print(f"本轮验证准确率{val_acc}")
        #把训练结果保存在当前目录下
        #当前py路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        torch.save(model.state_dict(), os.path.join(current_dir, "RnnLstmModel.pth"))
        #保存vocab
        torch.save(vocab, os.path.join(current_dir, "vocab.pth"))
    return vocab

def test_model(test_data):
    #评估模型
    #当前py路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #加载vocab
    vocab = torch.load(os.path.join(current_dir, "vocab.pth"))
    model = RnnLstmModel(vocab_size=len(vocab))
    #加载模型训练结果
    model.load_state_dict(torch.load(os.path.join(current_dir, "RnnLstmModel.pth")))
    #开启模型测试状态
    model.eval()
    ids = encode(test_data, vocab)
    x = torch.tensor([ids], dtype=torch.long) # 加上 batch 维度: [1, 5]
    
    #关闭梯度
    with torch.no_grad():
        y_p = model(x)
        # 多分类任务，取预测概率最大的索引
        pred_class = torch.argmax(y_p, dim=-1).item()
        print(f"输入文字: {''.join(test_data)}")
        print(f"模型预测的各位置分数: {y_p.tolist()[0]}")
        print(f"最终预测结果: '你' 字在第 {pred_class + 1} 个位置")

# 运行训练并获取词表
train()

# 运行测试
test = ["春", "夏", "秋", "你", "我"]
test_model(test)


