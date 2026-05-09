import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

# 样本数量
SAMPLES_NUM = 4000
# 句长与类别数一致：「你」的下标（第一个字为 0）→ 类别 0 … MAXLEN-1
MAXLEN = 5
MARK_CHINESE = "你"
SEED = 42
CATE_NUM = MAXLEN
TRAIN_RATIO = 0.8
BATCH_SIZE = 10
EMBED_DIM = 64
HIDDEN_DIM = 64
LR = 1e-3
EPOCHS = 10


random.seed(SEED)
def generate_random_chinese():
    while True:
        char_code = random.randint(0x4E00, 0x9FA5)
        ch = chr(char_code)
        if ch != MARK_CHINESE:
            return ch


def generate_random_sentence(num=0, mark_index=0):
    sentence = ""
    for i in range(num):
        if mark_index == i:
            sentence += MARK_CHINESE
        else:
            sentence += generate_random_chinese()
    return sentence


def category_from_sentence(sent):
    """「你」首次出现的字符下标 = 类别标签（第 1 个字→0，第 2 个字→1，…）。"""
    return sent.index(MARK_CHINESE)


def build_dataset(nums=0):
    data = []
    for _ in range(nums):
        pos_idx = random.randint(0, CATE_NUM - 1)
        sent = generate_random_sentence(MAXLEN, pos_idx)
        assert category_from_sentence(sent) == pos_idx
        data.append((sent, pos_idx))
    return data


def build_vocab(data):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids = [vocab.get(ch, 1) for ch in sent]
    ids += [0] * (maxlen - len(ids))
    return ids


class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.x = [encode(sent, vocab) for sent, _ in data]
        self.y = [cate for _, cate in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.x[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


class LstmClassifier(nn.Module):
    """中文句中「你」的位置分类：嵌入 + 双向 LSTM + 时间维 max pooling + 全连接。"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        lstm_out_dim = hidden_dim * 2
        self.bn = nn.BatchNorm1d(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, CATE_NUM)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        pooled = out.max(dim=1)[0]
        pooled = self.dropout(self.bn(pooled))
        return torch.softmax(self.fc(pooled), dim=1)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            y_prob = model(x)
            y_pred = torch.argmax(y_prob, dim=1)
            correct += (y_pred == y).sum().item()
            total += len(y)
    return correct / total


def test(model, vocab):
    cases = build_dataset(10)

    model.eval()
    print("\n--- BiLSTM 推理示例（真实 vs 预测）---")
    with torch.no_grad():
        for sent, true_idx in cases:
            x = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            pred_idx = torch.argmax(model(x), dim=1).item()
            hit = "正确" if pred_idx == true_idx else "错误"
            print(f"句子: {sent}")
            print(
                f"  真实: 「你」在下标 {true_idx}（第 {true_idx + 1} 个字）→ 类 {true_idx} | "
                f"预测: 类 {pred_idx} → {hit}"
            )


def train():
    data = build_dataset(SAMPLES_NUM)
    vocab = build_vocab(data)
    print(f"[BiLSTM] 样本数 {len(data)} 词表大小 {len(vocab)}")
    train_num = int(len(data) * TRAIN_RATIO)
    train_data = data[:train_num]
    val_data = data[train_num:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)
    model = LstmClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    test(model, vocab)


if __name__ == "__main__":
    train()
