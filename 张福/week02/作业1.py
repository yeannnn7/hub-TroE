
import os
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

    基于pytorch框架编写模型训练
    实现一个自行构造的找规律(机器学习)任务
    规律：x是一个5维向量，机器选择哪一维，就是哪一类
    --------------
    课堂上的作业,将现有的模型：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本 
    改为 ： x是一个5维向量，数据当中哪一列数据最大，就是哪一类

"""
class TorchModel(nn.Module):
    def __init__(self,input_size=5,output_size=20):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        # CrossEntropyLoss 内部自带了softmax激活函数,不需要单独调用.
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x,y=None):
        y_pred = self.linear(x)
        # print(y_pred,y)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            # return y_pred
            return torch.softmax(y_pred,dim=1) #输出标准的概率分布
# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，机器选择哪一维，就是哪一类
def build_sample():
    x = np.random.random(5)
    # print(x,np.argmax(x))
    return x,np.argmax(x)

# 随机生成一批样本
# 5类样本均匀生成
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() # 将模型设置为评估模式
    test_sample_num = 500
    x,y = build_dataset(test_sample_num)
    # for i in range(5):
    #     print(f"本次预测集中共有{(y==i).sum().item()}个{i+1}类")
    correct, wrong = 0, 0
    one,two,tree,four,five = 0,0,0,0,0
    with torch.no_grad():
        y_pred = model(x) # 获取预测结果（概率）
        # print(y_pred)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == int(y_t):
                correct += 1
                if int(y_t) == 0:
                    one +=1
                elif int(y_t) == 1:
                    two +=1
                elif int(y_t) == 2:
                    tree +=1
                elif int(y_t) == 3:
                    four +=1
                elif int(y_t) == 4:
                    five +=1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    print(f"""=======================
    0类识别正确个数:{one}/{(y==0).sum().item()}
    1类识别正确个数:{two}/{(y==1).sum().item()}
    2类识别正确个数:{tree}/{(y==2).sum().item()}
    3类识别正确个数:{four}/{(y==3).sum().item()}
    4类识别正确个数:{five}/{(y==4).sum().item()}
    """)
    return correct / (correct + wrong)

def main():
    #配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size,batch_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            # print(x,y)
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重 - 优化器
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    # print(log)
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
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        y_pred = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, y_p in zip(input_vec, y_pred):
        print("输入：%s, 预测类别：%d, 线性值：%f" % (vec, np.argmax(y_p) , y_p[np.argmax(y_p)]))  # 打印结果

if __name__ == '__main__':
    # main()
    # test()
    test_vec = [[0.88889086,0.95229675,0.31082123,0.03504317,0.88920843],
                [0.94963533,1.5524256,0.95758807,0.95520434,0.84890681],
                [0.90797868,0.67482528,2.13625847,0.34675372,0.19871392],
                [0.99349776,0.59416669,0.92579291,3.41567412,0.1358894],
                [1.99349776, 1.59416669, 1.92579291, 1.41567412, 2.0358894]
                ]

    predict("model.bin", test_vec)

