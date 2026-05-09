import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ======================
# 1. 超参数配置
# ======================

config = {
    "seq_len": 5,
    "train_size": 5000,
    "test_size": 1000,
    "batch_size": 64,
    "embed_dim": 32,
    "hidden_dim": 64,
    "num_layers": 1,
    "num_classes": 5,
    "learning_rate": 0.001,
    "epochs": 10,
    "seed": 42
}


# ======================
# 2. 固定随机种子
# ======================

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(config["seed"])


# ======================
# 3. 构建词表
# ======================

base_chars = list("我他她它们的是在有和中大人民学习天气快乐生活中国北京上海广州深圳苹果香蕉猫狗山水火木金土")
target_char = "你"

# 去重
base_chars = list(set(base_chars))

# 标准 NLP 词表
vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for ch in base_chars + [target_char]:
    if ch not in vocab:
        vocab[ch] = len(vocab)

char2idx = vocab
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(vocab)


# ======================
# 4. 数据生成
# ======================

def generate_sample():
    """
    生成一个长度为 5 的文本。
    文本中一定包含一个“你”字。
    标签为“你”所在的位置，取值为 0~4。
    """
    pos = random.randint(0, config["seq_len"] - 1)

    text = []
    for i in range(config["seq_len"]):
        if i == pos:
            text.append(target_char)
        else:
            text.append(random.choice(base_chars))

    return "".join(text), pos


class PositionDataset(Dataset):
    def __init__(self, size):
        self.samples = [generate_sample() for _ in range(size)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        text, label = self.samples[index]

        token_ids = [
            char2idx.get(ch, char2idx["<UNK>"])
            for ch in text
        ]

        x = torch.tensor(token_ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)

        return x, y


train_dataset = PositionDataset(config["train_size"])
test_dataset = PositionDataset(config["test_size"])

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["batch_size"],
    shuffle=False
)


# ======================
# 5. 定义模型
# ======================

class TextSequenceClassifier(nn.Module):
    def __init__(self, model_type="RNN"):
        super().__init__()

        self.model_type = model_type

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=config["embed_dim"],
            padding_idx=char2idx["<PAD>"]
        )

        if model_type == "RNN":
            self.rnn = nn.RNN(
                input_size=config["embed_dim"],
                hidden_size=config["hidden_dim"],
                num_layers=config["num_layers"],
                batch_first=True
            )

        elif model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=config["embed_dim"],
                hidden_size=config["hidden_dim"],
                num_layers=config["num_layers"],
                batch_first=True
            )

        elif model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=config["embed_dim"],
                hidden_size=config["hidden_dim"],
                num_layers=config["num_layers"],
                batch_first=True
            )

        else:
            raise ValueError("model_type must be one of: RNN, LSTM, GRU")

        self.fc = nn.Linear(
            config["hidden_dim"],
            config["num_classes"]
        )

    def forward(self, x):
        embedded = self.embedding(x)

        output, hidden = self.rnn(embedded)

        # 取最后一个时间步的输出
        last_output = output[:, -1, :]

        logits = self.fc(last_output)

        return logits


# ======================
# 6. 训练与评估函数
# ======================

def evaluate(model, data_loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            predictions = torch.argmax(logits, dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    return correct / total


def train_model(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"]
    )

    for epoch in range(config["epochs"]):
        model.train()

        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        test_acc = evaluate(model, test_loader, device)

        print(
            f"Epoch [{epoch + 1}/{config['epochs']}], "
            f"Loss: {avg_loss:.4f}, "
            f"Test Accuracy: {test_acc:.4f}"
        )

    return model


# ======================
# 7. 训练 RNN / LSTM / GRU
# ======================

results = {}

for model_type in ["RNN", "LSTM", "GRU"]:
    print("=" * 50)
    print(f"Training {model_type}")

    model = TextSequenceClassifier(model_type=model_type)
    trained_model = train_model(model, train_loader, test_loader)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_acc = evaluate(trained_model, test_loader, device)

    results[model_type] = final_acc


# ======================
# 8. 输出实验结果
# ======================

print("=" * 50)
print("Final Results")

for model_type, acc in results.items():
    print(f"{model_type}: {acc:.4f}")


# ======================
# 9. 简单预测示例
# ======================

def predict_text(model, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    token_ids = [
        char2idx.get(ch, char2idx["<UNK>"])
        for ch in text
    ]

    x = torch.tensor([token_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return pred


sample_text = "我爱你中"
print("=" * 50)
print(f"Input text: {sample_text}")
print(f"Predicted position of '你': {predict_text(trained_model, sample_text)}")
