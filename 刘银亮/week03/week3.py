import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

'''
Task Desc: 设计一个以文本为输入的多分类任务, 实验一下用 LSTM 模型的跑通训练
'''

# 定义最长文本长度
MAX_LEN = 40
# 定义训练轮数
EPOCHS = 20
# 定义训练集比例
TRAIN_RATIO = 0.8
# 定义设备(不支持 GPU)
DEVICE = "cpu"

# 读取数据集
def read_dataset():
    """
    数据集格式举例:
        还有双鸭山到淮阴的汽车票吗13号的	Travel-Query
        从这里怎么回家	Travel-Query
        随便播放一首专辑阁楼里的佛里的歌	Music-Play
        给看一下墓王之王嘛	FilmTele-Play
        我想看挑战两把s686打突变团竞的游戏视频	Video-Play
        我想看和平精英上战神必备技巧的游戏视频	Video-Play
        2019年古装爱情电视剧小女花不弃的花絮播放一下	Video-Play
    """
    dataset = pd.read_csv("/Users/mac/WorkSpace/Coding/Python/ai-practice/dataset.csv", sep="\t", header=None)
    texts = dataset[0].tolist()
    string_labels = dataset[1].tolist()

    # 构建 label index - label 映射
    label_list = sorted(set(string_labels))
    label_to_index = {label: i for i, label in enumerate(label_list)}
    numerical_labels = [label_to_index[label] for label in string_labels]

    # 构建词表
    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    index_to_char = {i: char for char, i in char_to_index.items()}
    vocab_size = len(char_to_index)
    return texts, vocab_size, char_to_index, index_to_char, numerical_labels

# 定义数据集类
class CharLSTMDataset(Dataset):
    def __init__(self, texts, numerical_labels, char_to_index, max_len):
        self.texts = texts
        self.numerical_labels = torch.tensor(numerical_labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.numerical_labels[idx]


# 定义模型
class LSTMClassifier(nn.Module):
    """
    中文关键词多分类器（LSTM）
    架构：Embedding → LSTM → BN → Dropout → Linear → Softmax → (CrossEntropyLoss)
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.bn = nn.LayerNorm1d(hidden_dim)
        # self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # bidirectional 时 hidden_dim * 2

    def forward(self, x):
        embedded = self.embedding(x)
        # LSTM 层
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # 拼接双向 LSTM 的最后隐藏状态: forward 最后一层 + backward 最后一层
        hidden_state = torch.cat((hidden_state[-2, :, :], hidden_state[-1, :, :]), dim=1)
        # hidden_state 形状: (batch_size, hidden_dim * 2) = (32, 512)
        out = self.dropout(hidden_state)  # 直接使用拼接后的 hidden state
        out = self.fc(out)
        return out

# 评估
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y).sum().item()
            total_samples += y.size(0)
    val_acc = total_correct / total_samples
    return total_loss / len(loader), val_acc
        
# 训练
def train():
    texts, vocab_size, char_to_index, index_to_char, numerical_labels = read_dataset()

    # 定义模型参数
    embedding_dim = 40
    hidden_dim = 256
    output_dim = len(set(numerical_labels))
    dropout = 0.3

    # 定义模型
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)
    model.to(DEVICE)

    #拆分训练集和验证集
    split = int(len(texts) * TRAIN_RATIO)
    train_texts = texts[:split]
    val_texts = texts[split:]
    train_labels = numerical_labels[:split]  # ✅ 分割标签
    val_labels = numerical_labels[split:]    # ✅ 分割标签

    print("num_classes:", output_dim)
    print("train size:", len(train_texts))

    # 定义训练集数据加载器
    dataloader = DataLoader(CharLSTMDataset(train_texts, train_labels, char_to_index, MAX_LEN), batch_size=32, shuffle=True)

    # 定义验证集数据加载器
    val_loader = DataLoader(CharLSTMDataset(val_texts, val_labels, char_to_index, MAX_LEN), batch_size=32, shuffle=True)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    loss_list = []
    accuracy_list = []
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        for x, y in dataloader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # 前向传播
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            # 反向传播和优化
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == y).sum().item()
            total_samples += y.size(0)

        train_acc = running_correct / total_samples
        # 验证模型
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        loss_list.append(val_loss)
        accuracy_list.append(val_accuracy)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 保存模型
    #torch.save(model.state_dict(), "lstm_model.pth")

    # 可视化
    visualize(loss_list, accuracy_list)

def visualize(loss_list, accuracy_list):
    plt.plot(loss_list, label="Val Loss")
    plt.plot(accuracy_list, label="Val Accuracy")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train()