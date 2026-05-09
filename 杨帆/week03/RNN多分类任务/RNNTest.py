import json
import random
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

"""
    字符串中”你“字第几个出现，就是第几类
"""

class RNNtest(nn.Module):
    def __init__(self, vocab, sentence_size, hidden_size):
        super(RNNtest, self).__init__()
        self.embedding = nn.Embedding(len(vocab), hidden_size, padding_idx=0)
        self.RNN = nn.RNN(hidden_size, hidden_size, bias=False, batch_first=True)
        self.classify = nn.Linear(hidden_size, sentence_size + 1)
        self.CrossEntropyLoss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, h = self.RNN(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_size)
        x = rnn_out[:, -1, :]  # (batch_size, sen_len, hidden_size) -> (batch_size, hidden_size)
        y_pred = self.classify(x)  # 将维度变为句子长度+1 (batch_size, hidden_size) -> (batch_size, sentence_size + 1)
        if y is not None:
            y_pred = self.CrossEntropyLoss(y_pred, y)
            return y_pred
        else:
            y_pred = nn.functional.softmax(y_pred, dim=-1)
            return y_pred

# 构造词表
def build_vocab():
    chars = "我爱你中国亲的母为流泪也自豪是谁从夜拽下一缕光将那已沉寂心扰乱明月高悬微风与星穿听哀叹红颜随了尘化作阑珊草长莺飞命落两三思却总常年漂泊花开日升没真情如烟波人只知寞在乎因果潮起圆看破不说太单薄能abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {'[pad]': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   #每个字对应一个序号
    vocab['[unk]'] = len(vocab)
    return vocab

# 构造样本
def build_sample(vocab, sen_size):
    x = [random.choice(list(vocab.keys())) for _ in range(sen_size)]
    if "你" in x:
        y = x.index("你")
    else:
        y = sen_size
    x = [vocab.get(word, vocab.get('[unk]')) for word in x]
    return x, y

# 构造数据集
def build_dataset(vocab, sample_num, sen_size):
    X = []
    Y = []
    for i in range(sample_num):
        x, y = build_sample(vocab, sen_size)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(np.asarray(X)), torch.LongTensor(np.asarray(Y))

# 训练代码
def train(batch_size):
    model.train()
    for batch_index in range(input_size // batch_size):
        watch_loss = []
        x = X[batch_index * batch_size : (batch_index + 1) * batch_size]
        y = Y[batch_index * batch_size : (batch_index + 1) * batch_size]
        optim.zero_grad()  # 梯度归零
        loss = model.forward(x, y)  # 计算loss
        loss.backward()  # 计算梯度
        optim.step()  # 更新权重
        watch_loss.append(loss.item())
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    return np.mean(watch_loss)

# 测试代码
def evaluate(model, vocab, sen_size):
    model.eval()
    test_input_size = 200
    x, y = build_dataset(vocab, test_input_size, sen_size)
    print("本次预测集中共有%d个样本" %test_input_size)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d，正确率：%f" %(correct, (correct / test_input_size)))
    return correct / test_input_size

# 预测代码
def predict(model_path, vocab_path, input_strings):
    sentence_size = 5  # 句子长度
    hidden_size = 128  # 隐向量维度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = RNNtest(vocab, sentence_size, hidden_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab["[unk]"]) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, np.argmax(result[i]).item(), result[i][:sentence_size + 1]))  # 打印结果

if __name__ == "__main__":
    sentence_size = 5  # 句子长度
    hidden_size = 128  # 隐向量维度
    epoch_num = 8  # 训练轮数
    batch_size = 2
    input_size = 700
    learning_rate = 0.001  # 学习率
    # 构造词表
    vocab = build_vocab()
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    model = RNNtest(vocab, sentence_size, hidden_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    X, Y = build_dataset(vocab, input_size, sentence_size)
    log = []
    for epoch in range(epoch_num):
        loss = train(batch_size)
        acc = evaluate(model, vocab, sentence_size)
        log.append([acc, float(loss)])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    strs = ["我爱你中国", "给你一杯茶", "你是我儿子", "这个是你的", "只有我和你"]
    predict('model.bin', "vocab.json", strs)

