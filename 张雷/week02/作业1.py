'''
尝试完成一个多分类任务的训练：一个随机向量，哪一维数字最大就属于第几类
基于pytorch框架编写模型训练
'''
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TorchModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        # 注意：CrossEntropyLoss内部包含了softmax，所以这里不需要额外加softmax
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (batch_size, num_classes) 输出logits
        if y is not None:
            return self.loss(x, y)  # 预测值和真实值计算损失
        else:
            # return x  # 输出logits
            return torch.softmax(x, dim=1)  # 输出概率分布:每行加起来等于1，表示每个类别的概率


# 生成一个样本，样本的生成方法，代表了我们要学习的规律
# 随机生成一个num_features维向量，哪一维数字最大就属于第几类
def build_sample(num_features):
    x = np.random.random(num_features)
    # 哪一维最大就属于第几类（0 ~ num_features-1）
    y = int(np.argmax(x))
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num, num_features):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample(num_features)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码，用来测试每轮模型的准确率
def evaluate(model, num_features):
    model.eval()
    test_sample_num = 20
    x, y = build_dataset(test_sample_num, num_features)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        _, predicted = torch.max(y_pred, 1)  # 取概率最大的类别
        print("预测类别：%s" % predicted)
        print("真实类别：%s" % y)
        total = y.size(0)  # 样本总数
        # correct = (predicted == y).sum().item() # 计算正确预测的个数
        for y_p, y_t in zip(predicted, y): # 逐个比较预测值和真实值
            if y_p == y_t: # 预测类别和真实类别相同
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / total))
    return correct / total


def main():
    # 配置参数
    epoch_num = 20          # 训练轮数
    batch_size = 64         # 每次训练样本个数
    train_sample = 5000     # 每轮训练总共训练的样本总数
    num_features = 10       # 输入向量维度
    num_classes = 10        # 分类数
    hidden_size = 50       # 隐藏层大小
    learning_rate = 0.001   # 学习率
    # 建立模型
    model = TorchModel(num_features, hidden_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample, num_features)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入 train_x[0:20]  train_y[0:20] 下一个batch train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()     # 计算梯度
            optim.step()        # 更新权重
            optim.zero_grad()   # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, num_features)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    num_features = 10
    num_classes = 10
    hidden_size = 50
    model = TorchModel(num_features, hidden_size, num_classes) # 创建模型实例
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    # print(model.state_dict()) # 打印模型权重，看看和训练时是否一样

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        print("模型输出：%s" % result)  # 打印模型输出的概率分布
        _, predicted = torch.max(result, 1)  # 取概率最大的类别
        print("预测类别：%s" % predicted)  # 打印预测类别
    for vec, pred in zip(input_vec, predicted):
        print("输入：%s, 预测类别：%d" % (vec, pred))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[1.88889086,0.15229675,0.31082123,0.03504317,0.88920843,0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,1.13625847,0.34675372,0.19871392,0.99349776,0.59416669,0.92579291,0.41567412,0.1358894],
                [0.09644111,0.08641935,0.64448924,0.98489686,0.12220344,1.07944809,0.04287028,0.99873789,0.76991342,0.4497638],
                [0.09644111,0.08641935,0.64448924,0.98489686,1.12220344,0.07944809,0.04287028,0.99873789,0.76991342,0.4497638],
                [0.09644111,0.08641935,0.64448924,0.98489686,0.12220344,0.07944809,0.04287028,0.99873789,0.76991342,0.4497638]]
    predict("model.bin", test_vec)
