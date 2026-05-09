"""
    作业,对字典进行分词。
    统计包含‘你’字的再哪一列，就是哪一类.
    设计一个以文本为输入的多分类任务，
    实验一下用RNN，LSTM等模型的跑通训练。
    如果不知道怎么设计，
    可以选择如下任务:
    对一个任意包含“你”字的五个字的文本，
    “你”在第几位，就属于第几类。
"""
import sys

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers.models.prophetnet.modeling_prophetnet import softmax

#1.读取分词数据包,建立单字字典映射关系


WORD_PAD = "[pad]"
WORD_UNK = "[unk]"
def read_dict():
    words = []
    with open('dict.txt',encoding='utf-8') as f:
        for _char in f.read().strip():
            # 去重
            if _char not in words and _char != '\n':
                words.append(_char)
    return words
    # print(word)
#2.建立单字字典映射关系
def build_words(words):
    with open('word.txt',mode='w',encoding='utf-8') as f:
        for word in  words:
            if word == words[len(words)-1]:
                f.write(word)
            else:
                f.write(word+'\n')
    pass
#3.词表构建与编码
def read_words():
    word2id = {WORD_PAD:0,WORD_UNK:1}
    with open('word.txt',mode='r',encoding='utf-8') as f:
        for line in f.readlines():
            for ch in line:
                if ch not in word2id and ch != '\n':
                    word2id[ch] = len(word2id)
        # vocab = [line.strip() for line in f.readlines()]
    # print('词表:',word2id)
    #构建词表
    # word2id = {word: idx for idx,word in enumerate(vocab)}
    # id2word = {idx: word for idx,word in enumerate(vocab)}
    return word2id
def build_embedding(words_len,embedding_dim=64):
    '''
    :param words_len: 词表大小
    :param embedding_dim: 向量维度 默认64维  通常64~512维
    :return:
    '''
    # 词表大小 * 向量维度
    embed = nn.Embedding(words_len, embedding_dim)
    print("Embedding...权重矩阵:\n",embed.weight)
    return embed
#将语句进行编码,将语句长度进行统一,对长短不一的进行补齐，不认识的未知，不够长的进行补齐
def sentence_encode(sentence,max_len,word2id):
    '''
    将语句进行编码,将语句长度进行统一,对长短不一的进行补齐，不认识的未知，不够长的进行补齐
    :param sentence:  语句
    :param max_len:   最长的语句长度
    :param word2id: 词表
    :return: 编码后的语句,转为词表id
    '''
    sentence_ids = []
    for word in sentence:
        if word in  word2id:
            sentence_ids.append(word2id[word])
        else:
            sentence_ids.append(word2id[WORD_UNK])
    sentence_ids += [word2id[WORD_PAD]]*(max_len-len(sentence))
    # while len(sentence_ids) < max_len:
    #     sentence_ids.append(word2id[WORD_PAD])
    return sentence_ids

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 40000
MAX_LEN     = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
#────────────────────────────────────────────────
FIND_ANSWER = '你'
# ─── 1. 数据生成 ────────────────────────────────────────────

#随机追加【你】
def make_random():
    sent = ''
    vocab = read_words()
    ran_int = random.randint(1,10) #随机1-10个文字
    for _ in range(ran_int):
        random_key = random.choice(list(vocab.keys()))
        sent+= random_key
    if random.random() < 0.3:
        extra = FIND_ANSWER
        pos   = random.randint(0, len(sent))
        #随机将【你】关键字加入到建立的数据当中
        sent  = sent[:pos] + extra + sent[pos:]
    return sent


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 2):
        # 随机样本
        pos_sent = make_random()
        pos_index = pos_sent.find(FIND_ANSWER) #如果没有找到则为-1
        pos_label = pos_index + 1  # 位置从0开始编号
        
        data.append((pos_sent, pos_label))
    random.shuffle(data)
    return data



# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [sentence_encode(s, MAX_LEN,vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # CrossEntropyLoss需要long类型标签
        )

# ─── 4. 模型定义 ────────────────────────────────────────────
class TorchModel(nn.Module):
    """
        中文关键词位置分类器（RNN + MaxPooling 版）
        架构：Embedding → RNN → MaxPool → BN → Dropout → Linear → Softmax → (CrossEntropyLoss)
        输出：预测"你"字在句子中的位置（0表示没有"你"，1~MAX_LEN表示具体位置）
    """
    def __init__(self,vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super(TorchModel, self).__init__()
        #词表大小 * 向量维度  , 默认将补全位设置为0
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # rnn 设置 向量维度 和 隐藏维度
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        # 归一化 对每个特征维度在 batch 方向归一化 → 稳定训练，加速收敛
        self.bn = nn.BatchNorm1d(hidden_dim)
        # 随机屏蔽 · 防过拟合 · 训练/推理切换
        self.dropout = nn.Dropout(p=dropout)
        # 输出层：MAX_LEN+1个类别（位置0到位置MAX_LEN）
        self.linear = nn.Linear(hidden_dim, MAX_LEN + 1)


    def forward(self, x):
        e, _ = self.rnn(self.embedding(x))  # (B, L, hidden_dim)
        # 池化,降维的作用
        pooled = e.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        # 归一化 + 随机屏蔽
        bn_dropout = self.dropout(self.bn(pooled))
        # 输出logits（CrossEntropyLoss会自动加softmax）
        y_pred = self.linear(bn_dropout)  # (B, MAX_LEN+1)
        return y_pred

# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval() #评估模式
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pre_y = model.forward(X)
            # 取概率最大的类别作为预测位置
            pre_y = torch.argmax(pre_y, dim=1)  # (B,)
            # 计算预测位置与真实位置完全匹配的数量
            correct += (pre_y == y).sum().item()
            total += len(y)
    return correct / total
def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = read_words()
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]
    # 封装数据集
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = TorchModel(vocab_size=len(vocab))  # 模型
    criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # 优化器

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()# 将模型设置为训练模式
        total_loss = 0.0
        for X, y in train_loader:
            pred = model.forward(X)
            loss = criterion(pred, y)  # CrossEntropyLoss：pred是logits，y是类别索引
            optimizer.zero_grad() # 梯度归0
            loss.backward() # 反向传播 -> 求导
            optimizer.step()# 更新损失函数参数 -> 会先去获取反向传播后的参数信息
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader) # 将训练好的模型 和 验证集 进行验证
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    # 保存模型
    torch.save(model.state_dict(), "model.bin")

# 使用训练好的模型做预测
def predict(model_path, test_sents):
    vocab = read_words() #用相同的词表,顺序就是训练的相同
    model = TorchModel(vocab_size=len(vocab))  # 模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval() #评估模式
    for sent in test_sents:
        ids = torch.tensor([sentence_encode(sent, MAX_LEN, vocab)], dtype=torch.long)
        logits = model.forward(ids)
        # dim=1 按行 选择  , dim=0 按列选择
        pred_pos = torch.argmax(logits, dim=1).item()
        #通过softmax 计算概率
        prob = torch.softmax(logits, dim=1)[0, pred_pos].item()
        if pred_pos == 0 or prob < 0.5:
            label = f"不含【你】字符 (概率:{prob:.2f})"
        else:
            label = f"【你】在位置 {pred_pos}(也就是类别) (概率:{prob:.2f})"
        print(f"  [{label}]  {sent}")

if __name__ == '__main__':
    # read_dict()
    # build_words()
    #--------------------------------
    # word2id = read_words()
    # print("word2id:",word2id)
    # embed = build_embedding(len(word2id))
    # #查询单个词向量
    # find_word = "你"
    # word_id = word2id[find_word]
    # print(f"{find_word} 的词向量:,",embed.weight.data[word_id])
    #----------------------------------
    # test_sentence_list = ["可接受的看见啊河口街道你","四u阿月覅u月份收到不","你你你111","263287364"]
    # for test_sentence in test_sentence_list:
    #     sentence_encode_ids = sentence_encode(test_sentence,MAX_LEN,word2id)
    #     print(sentence_encode_ids)
    #----------------------------------
    # train()
    #----------------------------------
    print("\n--- 推理示例 10个文字以内判定---")
    test_sents = [
        '你真的不错',
        '中国你好',
        '我你他',
        '就要告诉你',
        '就要你告诉',
        '嘿我就不告诉',
        '认真学习ai',
        '是保留前面信你息的方式',
        '是保留前你面信息的方式',
        '萨科技繁花盛开你就很烦',
    ]
    predict("model.bin", test_sents)





    pass