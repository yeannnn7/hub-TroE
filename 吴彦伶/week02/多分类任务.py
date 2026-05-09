# coding:utf8

# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务
规律：x是一个5维向量，哪一维数字最大就属于第几类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.loss = nn.functional.cross_entropy

    # 计算损失或者预测结果
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.softmax(y_pred, dim=1) #输出是一个概率分布，dim=1表示按行计算softmax
        
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律

# 随机生成一个5维向量
def build_sample():
    x = np.random.random(5)
    if x[0]>x[1] and x[0]>x[2] and x[0]>x[3] and x[0]>x[4]:
        return x, 0
    elif x[1]>x[0] and x[1]>x[2] and x[1]>x[3] and x[1]>x[4]:
        return x, 1
    elif x[2]>x[0] and x[2]>x[1] and x[2]>x[3] and x[2]>x[4]:
        return x, 2
    elif x[3]>x[0] and x[3]>x[1] and x[3]>x[2] and x[3]>x[4]:
        return x, 3
    else:
        return x, 4

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    X = np.array(X)  # 先转成单个 NumPy 数组
    Y = np.array(Y).squeeze()  # Y 也转成数组，并去掉多余维度
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, x_val, y_val):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x_val)  # 模型预测 model.forward(x)
        for y_p, y_t in zip(y_pred, y_val):  # 与真实标签进行对比
            if torch.argmax(y_p) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)
    
def main():
    #配置参数
    train_sample = 5000 #训练样本数量
    epoch_num = 100 #训练轮数
    batch_size = 20 #batch size

    # 1. 构造数据集
    X_train, Y_train = build_dataset(train_sample)

    # 2. 构造模型
    model = TorchModel(input_size=5)

    # 3. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    log = []

    # 4. 模型训练
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = X_train[batch_index*batch_size:(batch_index+1)*batch_size]
            y = Y_train[batch_index*batch_size:(batch_index+1)*batch_size]      
            #计算损失
            loss = model(x, y)
            #计算梯度
            loss.backward()
            #更新权重
            optimizer.step()
            #梯度清零
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("第%d轮， loss %f" %(epoch, np.mean(watch_loss)))
        if loss.item() < 0.00001:
            break
        acc = evaluate(model, X_train, Y_train)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "model.bin")                   
    #5. 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return    


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, torch.argmax(res), torch.max(res)))  # 打印结果

if __name__ == "__main__":
    main()
    test_vec = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    predict("model.bin", test_vec)
