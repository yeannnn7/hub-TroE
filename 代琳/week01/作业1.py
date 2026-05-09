
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""
多分类任务：
输入：随机生成的5维向量
规则：哪一维的数字最大，该样本就属于第几类（类别标签：0/1/2/3/4）
示例：[0.1, 0.8, 0.3, 0.5, 0.2] → 第1维（索引1）最大 → 标签为1

与二分类的核心区别：
- 二分类：输出1个值，用sigmoid + MSE/BCE
- 多分类：输出n个值（n=类别数），用softmax + CrossEntropyLoss
"""


# ===================== 1. 多分类模型定义 =====================
class TorchModel(nn.Module):
    """
    多分类神经网络模型
    结构：单层全连接网络 + CrossEntropyLoss（内置softmax）
    """

    def __init__(self, input_size, num_classes):
        """
        初始化函数

        参数：
            input_size: 输入特征维度 = 5
            num_classes: 分类类别数量 = 5（对应5个类别：0,1,2,3,4）
        """
        super(TorchModel, self).__init__()

        # 线性层：将5维输入映射到5维输出（每个维度对应一个类别的得分）
        # 内部参数：
        #   weight: shape (5, 5)，5×5的权重矩阵
        #   bias: shape (5,)，5个偏置项
        # 输出：每个样本得到5个"未归一化的得分"（logits）
        self.linear = nn.Linear(input_size, num_classes)  # (batch, 5) → (batch, 5)

        # 多分类专用损失函数：交叉熵损失
        # 重要特性：
        #   1. 内置softmax操作，不需要手动添加
        #   2. 标签必须是LongTensor类型的整数（0,1,2,3,4）
        #   3. 会自动计算softmax后再计算负对数似然损失
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        前向传播函数

        参数：
            x: 输入数据，shape为 (batch_size, 5)
            y: 真实标签，shape为 (batch_size,)，训练时提供，预测时为None

        返回：
            训练时：返回loss值（标量）
            预测时：返回概率分布，shape为 (batch_size, 5)
        """
        # 第一步：线性变换，得到每个类别的原始得分（logits）
        # 输入：(batch_size, 5)
        # 输出：(batch_size, 5)，例如 [[2.1, -0.5, 1.3, 0.8, -1.2], ...]
        # 注意：这里的值没有经过softmax，可能为负数，也可能大于1
        x = self.linear(x)

        # 如果提供了真实标签，说明是训练模式，计算损失
        if y is not None:
            # CrossEntropyLoss会自动执行：
            #   1. softmax：将logits转为概率分布
            #   2. 计算交叉熵损失
            # 所以这里不需要手动调用softmax！
            return self.loss(x, y)
        else:
            # 预测模式：手动应用softmax，将得分转为概率分布
            # dim=1 表示对每一行（每个样本）独立进行softmax
            # 例如：[2.1, -0.5, 1.3, 0.8, -1.2] → [0.45, 0.04, 0.20, 0.15, 0.06]
            # 所有概率之和 = 1.0
            return torch.softmax(x, dim=1)


# ===================== 2. 生成多分类样本 =====================
def build_sample(input_size=5):
    """
    生成单个样本

    参数：
        input_size: 向量维度，默认5

    返回：
        x: 随机生成的5维向量，每个值在[0,1)之间
        y: 标签（整数），表示最大值所在的索引位置

    示例：
        生成 x = [0.15, 0.82, 0.33, 0.51, 0.07]
        最大值是 0.82，在索引1的位置
        返回 (x, 1)
    """
    x = np.random.random(input_size)  # 生成5个[0,1)的随机数
    y = np.argmax(x)  # 找出最大值的索引位置（0~4之间的整数）
    return x, y


def build_dataset(total_sample_num, input_size=5):
    """
    构建完整数据集

    参数：
        total_sample_num: 需要生成的样本总数
        input_size: 每个样本的特征维度

    返回：
        X: 特征矩阵，shape (total_sample_num, 5)，FloatTensor类型
        Y: 标签向量，shape (total_sample_num,)，LongTensor类型

    重要说明：
        - 特征用FloatTensor（浮点数）
        - 标签必须用LongTensor（长整型），这是CrossEntropyLoss的要求
        - 标签不需要one-hot编码，直接用类别索引（0,1,2,3,4）
    """
    X, Y = [], []

    for _ in range(total_sample_num):
        x, y = build_sample(input_size)
        X.append(x)
        Y.append(y)  # 注意：这里直接append整数y，不需要[y]

    # 转换为PyTorch张量
    # X: [[0.15, 0.82, ...], [0.33, 0.12, ...], ...] → shape (N, 5)
    # Y: [1, 2, 0, 4, ...] → shape (N,)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签必须是LongTensor！


# ===================== 3. 多分类评估函数 =====================
def evaluate(model, input_size=5):
    """
    评估模型在测试集上的准确率

    参数：
        model: 待评估的模型
        input_size: 特征维度

    返回：
        acc: 准确率（0~1之间的浮点数）
    """
    model.eval()  # 切换到评估模式（关闭dropout等）

    test_sample_num = 100  # 测试样本数量
    x, y = build_dataset(test_sample_num, input_size)  # 生成100个测试样本

    correct = 0  # 初始化正确预测计数器

    with torch.no_grad():  # 评估时不计算梯度，节省内存
        # 模型预测，得到概率分布
        # y_pred shape: (100, 5)，每行5个概率值，和为1
        y_pred = model(x)

        # 取概率最大的类别作为预测结果
        # torch.argmax(y_pred, dim=1) 对每一行取最大值索引
        # 例如：[[0.1, 0.7, 0.1, 0.05, 0.05], ...] → [1, ...]
        pred_classes = torch.argmax(y_pred, dim=1)  # shape: (100,)

        # 比较预测类别和真实类别
        # pred_classes == y 返回布尔张量：[True, False, True, ...]
        # .sum() 计算True的个数
        # .item() 转换为Python整数
        correct = (pred_classes == y).sum().item()

    # 计算准确率
    acc = correct / test_sample_num

    print(f"正确预测：{correct}/{test_sample_num}，准确率：{acc:.4f}")
    return acc


# ===================== 4. 训练主函数 =====================
def main():
    """完整的训练流程"""

    # ========== 配置超参数 ==========
    epoch_num = 20  # 训练轮数：遍历整个数据集20次
    batch_size = 20  # 批大小：每次训练20个样本
    train_sample = 5000  # 每轮训练样本总数
    input_size = 5  # 输入特征维度
    num_classes = 5  # 分类类别数（5分类）
    learning_rate = 0.01  # 学习率：控制参数更新步长

    # ========== 初始化模型和优化器 ==========
    model = TorchModel(input_size, num_classes)  # 创建模型

    # Adam优化器：自适应学习率，收敛更快
    # model.parameters()获取所有可训练参数（weight和bias）
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ========== 准备训练数据 ==========
    # train_x: shape (5000, 5)，5000个5维向量
    # train_y: shape (5000,)，5000个标签（0~4的整数）
    train_x, train_y = build_dataset(train_sample, input_size)

    log = []  # 记录每轮的[准确率, 平均loss]，用于绘图

    # ========== 开始训练循环 ==========
    for epoch in range(epoch_num):
        model.train()  # 切换到训练模式
        total_loss = []  # 记录当前轮次每个batch的loss

        # 分批训练：5000个样本 / 每批20个 = 250个batch
        for i in range(train_sample // batch_size):
            # 取出当前batch的数据（切片）
            # 第0批：[0:20], 第1批：[20:40], ..., 第249批：[4980:5000]
            x = train_x[i * batch_size: (i + 1) * batch_size]  # shape: (20, 5)
            y = train_y[i * batch_size: (i + 1) * batch_size]  # shape: (20,)

            # 四步训练流程：
            loss = model(x, y)  # 1. 前向传播 + 计算loss
            loss.backward()  # 2. 反向传播，计算梯度
            optim.step()  # 3. 更新权重参数
            optim.zero_grad()  # 4. 梯度清零（防止累积）

            total_loss.append(loss.item())  # 记录当前batch的loss值

        # 打印当前轮次的平均loss
        # np.mean(total_loss) 计算250个batch的平均loss
        print(f"第{epoch + 1}轮 loss：{np.mean(total_loss):.4f}")

        # 评估当前模型的准确率
        acc = evaluate(model, input_size)

        # 记录本轮结果
        log.append([acc, np.mean(total_loss)])

    # ========== 保存训练好的模型 ==========
    # 只保存模型参数（weight和bias），不保存模型结构
    # 文件后缀.bin表示二进制文件
    torch.save(model.state_dict(), "multi_model.bin")

    print("\n模型已保存到 multi_model.bin")

    # ========== 可视化训练过程 ==========
    # 绘制准确率曲线和loss曲线
    plt.figure(figsize=(10, 5))

    # 第一条曲线：准确率（绿色）
    plt.plot([l[0] for l in log], label="accuracy", marker='o', color='green')

    # 第二条曲线：loss（红色）
    plt.plot([l[1] for l in log], label="loss", marker='x', color='red')

    plt.xlabel('Epoch')  # x轴标签
    plt.ylabel('Value')  # y轴标签
    plt.title('Training Process')  # 图表标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图表


# ===================== 5. 使用训练好的模型进行预测 =====================
def predict(model_path, input_vec):
    """
    加载训练好的模型，对新样本进行预测

    参数：
        model_path: 模型文件路径（.bin文件）
        input_vec: 待预测的输入向量列表，shape (N, 5)

    示例：
        test_vec = [[0.1, 0.7, 0.2, 0.1, 0.0],
                    [0.3, 0.2, 0.9, 0.1, 0.2]]
        predict("multi_model.bin", test_vec)
    """
    input_size = 5
    num_classes = 5

    # 创建模型实例（需要和训练时的结构完全一致）
    model = TorchModel(input_size, num_classes)

    # 加载训练好的权重参数
    # state_dict()返回：{'linear.weight': tensor, 'linear.bias': tensor}
    model.load_state_dict(torch.load(model_path))

    model.eval()  # 切换到评估模式

    # 转换为tensor并进行预测
    # input_vec: list of lists → FloatTensor: shape (N, 5)
    with torch.no_grad():  # 不计算梯度
        # 模型预测，也可写作model.forward(torch.FloatTensor(input_vec))
        prob = model(torch.FloatTensor(input_vec))  # 得到概率分布，shape (N, 5)
        pred_cls = torch.argmax(prob, dim=1)  # 取概率最大的类别，shape (N,)

    # 打印每个样本的详细预测结果
    for vec, p, cls in zip(input_vec, prob, pred_cls):
        print("=" * 50)
        print(f"输入向量：{vec}")
        print(f"各类概率：{p.numpy()}")  # 转为numpy数组方便查看
        print(f"预测类别：第{cls.item()}类")
        print(f"最高概率：{p[cls].item():.4f}")

        # 验证：找出输入向量中最大的维度
        true_class = np.argmax(vec)
        print(f"真实类别：第{true_class}类（基于输入向量最大值位置）")

        if cls.item() == true_class:
            print("✓ 预测正确！")
        else:
            print("✗ 预测错误！")
        print()


if __name__ == "__main__":
    # 运行训练（取消注释）
    # main()

    # 测试预测功能
    # 构造3个测试样本
    test_vec = [
        [0.1, 0.7, 0.2, 0.1, 0.0],  # 最大值在索引1 → 应该是第1类
        [0.3, 0.2, 0.9, 0.1, 0.2],  # 最大值在索引2 → 应该是第2类
        [0.7, 0.1, 0.1, 0.1, 0.0]  # 最大值在索引0 → 应该是第0类
    ]

    print("开始预测...")
    predict("multi_model.bin", test_vec)
