import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class MultiClassficationModel(nn.Module):
    def __init__(self, input_size):
        super(MultiClassficationModel, self).__init__()
        self.liner = nn.Linear(input_size, 5) # 线性层
        self.loss = nn.functional.cross_entropy # 交叉熵计算损失loss

    # 自定义函数，当输入真实标签返回loss，当无真实值返回预测值
    def forward(self, x,y=None):
        y_pred = self.liner(x)
        if y is not None:
            return self.loss(y_pred, y) # 预测值与真实值计算损失loss
        else:
            return torch.softmax(y_pred, dim=1) # softmax输出预测值

# 随机生成一个6维向量，根据每个向量中最大的标量同一下标构建Y
def build_sample():
    x = np.random.random(5)
    # 获取最大值的索引
    max_index = np.argmax(x)
    return x, max_index

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    succsee, faild = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测
        for y_p, y_t in zip(y_pred, y):
            # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                succsee += 1
            else:
                faild += 1
    print("正确预测个数：%d, 正确率：%f" % (succsee, succsee / (succsee + faild)))
    return succsee / (succsee + faild)

def main():
    # 配置参数
    epoch_num = 100 # 训练轮次
    batch_size = 20 # 每个批次训练样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5 # 向量维度
    learning_rate = 0.005 # 学习率

    # 建立模型
    model = MultiClassficationModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y) # 计算loss model.forward(x, y)
            loss.backward() # 计算梯度
            optim.step() # 更新权重
            optim.zero_grad # 梯度清零，可以在计算梯度之前做
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.pt")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 调用已训练的模型
def predict(model_path, input_test):
    input_size = 5
    model = MultiClassficationModel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        result = model.forward(torch.FloatTensor(input_test)) # 模型预测
    for vec, res in zip(input_test, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == '__main__':
    main()
    test = [[0.55, 0.478, 0.15, 0.33, 0.03],
            [0.11, 0.52, 0.77, 0.44, 0.33],
            [0.15 ,0.113, 0.565, 0.335, 0.31],
            [0.12, 0.23, 0.34, 0.43, 0.32,],
            [0.012, 0.513, 0.336, 0.03, 0.62,],
            [0.042, 0.025, 0.351, 0.265, 0.568]]
    predict("model.pt", test)
