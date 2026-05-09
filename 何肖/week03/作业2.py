"""
学员姓名：何肖
"""
import random 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os

"""
N第三周作业作业:设计一个以文本为输入的多分类任务，
实验一下用RNN模型的跑通训练。
如果不知道怎么设计，可以选择如下任务:对一个任意包含“你”字的不长于32个字的文本，
“你”在第几位，就属于第几类。
"""
# 定义超参数
SEED        = 42
MAXLEN      = 32
EMBED_DIM   = 64
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 40
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

def build_dataset():
    """
    构建数据集，包含文本和对应的类别
    """
    # 获取当前脚本所在的文件夹路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 sentence.txt 的完整路径
    sentence_path = os.path.join(script_dir, "update_2000_sentences.txt")

    with open(sentence_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    data = []

    for line in lines:
        for i, ch in enumerate(line):
            if ch == "你":
                data.append((line, i))            
    return data


def build_vacab(data):
    """
    构建词汇表，包含PAD和unk
    :param data: 数据集，每个样本为 (文本, 类别)
    :return: 词汇表，键为字符，值为索引
    """
    vacab = {'<PAD>': 0, "<unk>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vacab:
                vacab[ch] = len(vacab)
    return vacab

def encode(sent, vacab, maxlen=MAXLEN):
    """
    对文本进行编码，将每个字符转换为对应的词汇表索引
    """
    ids = [vacab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [vacab['<PAD>']] * (maxlen - len(ids))
    return ids

# Dataset / DataLoader
class TextDataset(Dataset):
    """
    自定义数据集类，用于加载文本数据
    TextDataset 通过继承 Dataset，获得了作为 PyTorch 数据集所需的所有基本功能。
    """
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [idx for _, idx in data]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long)
    )


# 模型定义
class multiClassRnn(nn.Module):
    """
    多分类RNN模型，用于文本分类任务
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 32)
        self.lose = nn.CrossEntropyLoss()

    # 前向传播，计算损失
    def forward(self, x, y=None):
        """
        前向传播，计算损失
        """
        x = self.embedding(x) #输出batch_size, seq_lenth, embed_dim=64
        e, _ = self.rnn(x) #输出batch_size, seq_lenth, hidden_dim*2=128
        pooled = e.max(dim=1)[0] # #输出(batch_size, hidden_dim*2=128)
        pooled = self.dropout(pooled) #输出batch_size, hidden_dim*2=128
        y_pred = self.fc(pooled) #  映射到32个类别
       
        if y is not None:
            lose = self.lose(y_pred, y.long())
        else:
            lose = None  
        return y_pred, lose

def evaluate(model, loader):
    """
    评估模型在验证集上的准确率
    """
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        for X, y in loader:
            y_pred_tensor, _ = model(X)
            pred_label = torch.argmax(y_pred_tensor, dim=1)

            for pred_label, true_labels in zip(pred_label, y):
                if pred_label == true_labels.item():
                    correct += 1
                else:
                    wrong += 1
    accuracy = correct / (correct + wrong)
    return accuracy

def train():
    """
    训练模型
    """
    print("生成数据集...")
    data = build_dataset()
    vocab = build_vacab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    # 数据加载器
    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE, shuffle=False) 

    # 模型实例化
    model = multiClassRnn(vocab_size=len(vocab))
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel()for p in model.parameters())
 
    print(f" 模型参数量: {total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_lose = 0.0

        for X, y in train_loader:
            pred, lose = model(X, y)
            optimizer.zero_grad() # 梯度清零
            lose.backward() # 反向传播，
            optimizer.step() # 改模型参数，让下一次更准
            total_lose += lose.item() # 计算损失
        
        avg_lose = total_lose / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS} lose={avg_lose:.4f} val_acc={val_acc:.4f}")

    # 保存模型
    print("模型已保存")
    torch.save(model.state_dict(), "model.pth")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()

    test_sents = [
    "清晨你看见新的答案",      # 你的位置: 2
    "路上听见你熟悉的歌",      # 你的位置: 4
    "雨后想起温暖回忆你",      # 你的位置: 8
    "你在海边等待下一站",      # 你的位置: 0
    "公园我把美好消息给你",    # 你的位置: 9
    ]

    with torch.no_grad():
        for sent in test_sents:
            ids = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            y_pred_tensor, _ = model(ids)
            pred_label = torch.argmax(y_pred_tensor, dim=1) 
            print(f"{sent}: {pred_label.item()}")

if __name__ == '__main__':
    train()
                               
               







            




       




       