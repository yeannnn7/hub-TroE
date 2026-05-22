# coding:utf8

# 解决 OpenMP 库冲突问题
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):
    def __init__(self, input_size,hidden_dim,num_classes):
        super(TorchModel, self).__init__()
        # self.linear = nn.Linear(input_size, 1)  # 线性层
        # self.activation = torch.sigmoid  # nn.Sigmoid() sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  # loss函数采用均方差损失
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x):
        return self.net(x)


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample(num_samples, dim=10):
    x = torch.randn(num_samples, dim)  # 随机向量
    y = torch.argmax(x, dim=1)         # 最大值索引作为标签
    return x, y


# 随机生成一批样本
# 正负样本均匀生成
# 超参数
INPUT_DIM = 10
NUM_CLASSES = 10
HIDDEN_DIM = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 50

# 生成训练集和测试集
X_train, y_train = build_sample(2000, INPUT_DIM)
X_test, y_test = build_sample(1000, INPUT_DIM)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


print(f"训练集: {len(train_dataset)} 样本")
print(f"测试集: {len(test_dataset)} 样本")


model = TorchModel(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X, y in loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    
    return total_loss / total, correct / total

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:  # 与真实标签进行对比
           y_pred = model(x)  # 模型预测 model.forward(x)
           loss = criterion(y_pred , y )
           total_loss += loss.item() * x.size(0)
           _, predicted = torch.max(y_pred, 1)
           total += y.size(0)
           correct += (predicted == y).sum().item()

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / total))
    return total_loss / total, correct / total

def main():
   # 用于存储历史数据
  history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
  }
    # 训练过程
  for epoch in range(1, EPOCHS + 1):
        train_loss , train_acc = train_epoch(model, test_loader, criterion , optimizer)  # 测试本轮模型结果
        test_loss , test_acc = evaluate(model, test_loader, criterion)  # 测试本轮模型结果
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    # 保存模型
        torch.save(model.state_dict(), "model.bin")
    # 画图
  print(history['train_loss'])

  epochs_range = range(1, EPOCHS + 1)
  plt.plot(epochs_range, history['train_loss'], label='Train Loss', linewidth=2)  # 画acc曲线
  plt.plot(epochs_range, history['test_loss'], label='Test Loss', linewidth=2)  # 画loss曲线
  plt.plot(epochs_range, history['train_acc'], label='Train Acc', linewidth=2)  # 画acc曲线
  plt.plot(epochs_range, history['test_acc'], label='Test Acc', linewidth=2)  # 画loss曲线
  plt.legend()
  plt.show()
  return
 

# 使用训练好的模型做预测
def predict(model_path):
    input_size = torch.randn(5 , INPUT_DIM)
    sample_output = model(input_size)
    sample_pred = torch.argmax(sample_output, dim=1)
    sample_true = torch.argmax(input_size, dim=1)
    for i in range(5):
        vec = input_size[i].numpy()
        max_idx = sample_true[i].item()
        pred_idx = sample_pred[i].item()
        correct_mark = "✓" if max_idx == pred_idx else "✗"
        print(f"样本{i+1} | 真实: {max_idx} | 预测: {pred_idx} {correct_mark}")
        print(f"  向量片段: [{vec[0]:.2f}, {vec[1]:.2f}, ...] (最大值: {vec[max_idx]:.2f})")


if __name__ == "__main__":
    main()
    model.eval()
    predict("model.bin")
