# coding:utf8
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
训练用 0~9
输出显示 1~10
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, class_num):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, class_num)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 训练用 0~9
        else:
            return torch.argmax(y_pred, dim=1) + 1  # 预测 +1 → 1~10

# 生成样本：训练标签 0~9
def build_sample():
    x = np.random.random(10)
    max_index = x.argmax()  # 0~9
    return x, max_index

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)

# 测试：模型输出是1~10，真实标签也要+1对比
def evaluate(model):
    model.eval()
    x, y = build_dataset(100)
    correct = 0
    with torch.no_grad():
        y_pred = model(x)
    for y_p, y_t in zip(y_pred, y):
        if y_p == y_t + 1:
            correct += 1
    print("正确：%d, 正确率：%f" % (correct, correct / 100))
    return correct / 100

def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 10
    class_num = 10
    learning_rate = 0.01

    model = TorchModel(input_size, class_num)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_x, train_y = build_dataset(train_sample)
    log = []

    for epoch in range(epoch_num):
        model.train()
        losses = []
        for i in range(train_sample // batch_size):
            x = train_x[i*batch_size : (i+1)*batch_size]
            y = train_y[i*batch_size : (i+1)*batch_size]
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
        print("第%d轮 loss:%.4f" % (epoch+1, np.mean(losses)))
        acc = evaluate(model)
        log.append([acc, np.mean(losses)])

    torch.save(model.state_dict(), "model.bin")
    plt.plot([l[0] for l in log], label="acc")
    plt.plot([l[1] for l in log], label="loss")
    plt.legend()
    plt.show()

def predict(model_path, input_vec):
    model = TorchModel(10, 10)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        res = model(torch.FloatTensor(input_vec))
    for vec, r in zip(input_vec, res):
        print(f"输入：{vec} → 分类为:{r.item()}")

if __name__ == "__main__":
    main()
    test_vec = [
        [1,2,3,4,5,6,7,8,9,10],    # 应输出 10
        [1,11,3,4,5,6,7,8,9,10],   # 应输出 2
        [1,2,3,12,5,6,7,8,9,10],   # 应输出 4
        [1,2,3,4,13,6,7,8,9,10]    # 应输出 5
    ]
    predict("model.bin", test_vec)