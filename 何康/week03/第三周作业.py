'''
设计一个以文本为输入的多分类任务，实验一下用RNN，LSTM等模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。

eg:三分类
文本       类别
你好啊       0
好你啊       1
啊好你       2
'''

import torch
import torch.nn as nn
import torch.optim as optim #优化器
import random

chars = ['你', '我', '他', '她', '好', '吗', '啊']
char2id = {c: i for i, c in enumerate(chars)} #给数字编号

def generate_sample(batch_size = 32): 
    '''
    text = random.choices(chars, k = 5) #五分类
    pos = random.randint(0, 4)
    text[pos] = '你'
    label = pos #你的位置
    return text, label
    '''

    texts = []
    labels = []
    for _ in range(batch_size):
       text = random.choices(chars, k = 5)
       pos = random.randint(0, 4)
       text[pos] = '你'

       texts.append([char2id[c] for c in text])
       labels.append(pos)

    return torch.tensor(texts), torch.tensor(labels)

def encode(text):
    return [char2id[c] for c in text] #把文本转换成数字

# 定义模型
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes): # 字符数量， 每个字符变成几维向量， 隐藏层维度， 分类数
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size) #把字符转换成向量
        #self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True) #输入维度， 隐藏层维度
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True) #输入维度， 隐藏层维度
        self.fc = nn.Linear(hidden_size, num_classes) #全连接层

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x) #每个时间步的输出
        out = out[:, -1, :] #取最后一个时间步的1输出
        out = self.fc(out)

        return out
    
model = RNNModel(vocab_size=len(chars), embed_size=8, hidden_size=16, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01) #更新参数模型

for epoch in range(200):

    '''
    text, label = generate_sample()
    x = torch.tensor([encode(text)])
    if epoch % 200 == 0:
        print(x.shape)
    y = torch.tensor([label])
    '''
    x, y = generate_sample(32)

    output = model(x) #前向传播
    loss = criterion(output, y) #计算损失

    optimizer.zero_grad() #清空梯度
      
    loss.backward() #反向传播

    optimizer.step() #更新参数

    if epoch % 20 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

# 测试
test1 = ['我', '好', '你', '啊', '他']
test2 = ['你', '他']
test3 = ['我', '你', '她', '她', '她']
#x = torch.tensor([encode(test_text)])
x = torch.tensor([[char2id[c] for c in test3]])
print(x)

model.eval()
with torch.no_grad():
    pred = model(x).argmax(dim = 1).item()

print("预测位置：", pred)
