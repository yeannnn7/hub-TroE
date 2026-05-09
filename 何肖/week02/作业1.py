"""
学员姓名：何肖
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
实现一个5分类模型，输入一个随机向量，哪一位数字最大就属于第几类
"""
class TorchModel(nn.Module):
    """
    5分类模型
    """
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 线性层, 负责计算
        #采用交叉熵lose函数，交叉熵适合多分类问题
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        前向传播，计算损失
        """
        # (batch_size, input_size) -> (batch_size, 5)
        y_pred = self.linear(x) 
        if y is not None:
            loss = self.loss(y_pred, y.long())
        else:
            loss = None
        return y_pred, loss
        

def build_sample():
   """
   生成一个样本
   """
   x = np.random.random(5)
   y = int(np.argmax(x))  # 返回最大值的索引
   return x, y

def build_dataset(total_sample_num):
    """
    生成数据集，包含total_sample_num个样本
    """

    # 创建空列表，存储样本和标签标签
    X = []
    Y = []
    
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    
    # 转换为张量并返回
    return torch.FloatTensor(X), torch.LongTensor(Y)

def evaluate(model):
    """
    评估模型准确率
    """
    model.eval() # 开启评估模式
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num) # 建立数据集

    correct, wrong = 0, 0 # 初始化正确和错误的数值
    with torch.no_grad():
        y_pred_tensor, _ = model(x) # 前向传播，计算预测结果

        predicated_labels = torch.argmax(y_pred_tensor, dim=1)  # 取每行最大值的索引，得到预测的类别标签

        # 计算正确率和错误率的数值
        for pred_label, true_labels in zip(predicated_labels, y):   
            if pred_label == true_labels.item():
                correct += 1
            else:     
                wrong += 1
    accuracy = correct / (correct + wrong)
    
    print("模型在测试集上正确个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size)

    # 建立优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 记录训练过程中的acc和lose
    log = []

    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练模型
    for epoche in range(epoch_num):
        model.train() # 开启训练模式
        watch_lose = [] # 记录当前轮的lose
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]
            _, loss = model(x, y) # 前向传播，计算损失
            loss.backward() # 反向传播，计算梯度
            optim.step() # 根据梯度更新参数 
            optim.zero_grad() # 清空梯度
            watch_lose.append(loss.item()) # 记录当前batch的lose
        print("=========\n第%d轮平均lose:%f" % (epoche + 1, np.mean(watch_lose)))
        acc = evaluate(model) # 评估模型准确率
        log.append([acc, float(np.mean(watch_lose))]) # 计算当前轮的准确率和错误率
        
    # 保存模型
    torch.save(model.state_dict(), "model.pth")

    # 打印模型权重
    print("\n模型权重参数：")
    for name, param in model.named_parameters():
        print(f"参数名称: {name}")
        print(f"参数值: {param.data}")
        print(f"参数形状: {param.shape}")
        print("-" * 40) 
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return 

def predict(model_path, input_vec):
    """
    评估模型预测类别
    """

    input_size = 5  # 输入向量维度
    model = TorchModel(input_size) # 建立模型
    model.load_state_dict(torch.load(model_path)) # 加载训练好的模型
    print(model.state_dict()) # 打印模型权重

    model.eval() # 开启评估模式
    
    with torch.no_grad():
        # 前向传播，计算预测结果
        y_pred, _ = model.forward(torch.FloatTensor(input_vec)) 

    # 计算概率值
        probabilities = torch.softmax(y_pred, dim=1)
        predicted_classes = torch.argmax(y_pred, dim=1)

    for vec, pred_class, prob_dist in zip(input_vec, predicted_classes, probabilities): 
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, pred_class.item(), prob_dist[pred_class].item()))  # 打印结果

if __name__ == "__main__":

    main()
    test_vec = [
    [0.1, 0.2, 0.3, 0.4, 0.5],       # 最大在4 → 标签4
    [1.0, 0.9, 0.8, 0.7, 0.6],       # 最大在0 → 标签0
    [0.5, 0.6, 0.7, 0.8, 0.7],       # 最大在3 → 标签3
    [0.4, 0.5, 0.1, 0.2, 0.3],       # 最大在1 → 标签1
    [0.2, 0.1, 0.9, 0.3, 0.4],       # 最大在2 → 标签2
    [0.9, 0.1, 0.1, 0.1, 0.1],       # 最大在0 → 标签0
    [0.1, 0.8, 0.2, 0.2, 0.2],       # 最大在1 → 标签1
    [0.1, 0.1, 0.7, 0.3, 0.2],       # 最大在2 → 标签2
    [0.2, 0.3, 0.2, 0.6, 0.1],       # 最大在3 → 标签3
    [0.1, 0.2, 0.3, 0.4, 0.9],       # 最大在4 → 标签4
    ]
    predict("model.pth", test_vec) 