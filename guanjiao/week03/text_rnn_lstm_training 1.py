# coding: utf-8
"""
文本多分类训练任务：
输入任意 5 个字的文本，文本中必须包含“你”字。
“你”在第几个位置，文本就属于第几类。

例如：
    "你真好呀哈" -> 第 1 类
    "我你真好呀" -> 第 2 类
    "我们你真好" -> 第 3 类
"""

import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# 解决中文输出乱码问题
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 文本长度
SEQ_LEN = 5
# 类别数量
CLASS_NUM = 5
# “你”字
NI_CHAR = "你"
# 输出目录
OUTPUT_DIR = Path(__file__).resolve().parent / "text_class_models"


# 这里准备一些普通汉字，用来随机拼出 5 个字的训练文本。
# 因为根据“你”的位置，来确定文本属于哪一类，所以字符池里不能包含“你”。避免无法确定类别。
CHAR_POOL = list("我他她它好爱学人工智能天气快乐努力编程数据模型训练分类输入输出北京上海广州深圳6谈五行八卦个方法")

# 建立字符表
def build_vocab():
    """建立字符表，把每个字转成模型可以处理的数字编号。"""
    vocab = ["<UNK>", NI_CHAR] + sorted(set(CHAR_POOL))
    return {char: index for index, char in enumerate(vocab)}


def encode_text(text, char_to_id):
    """把文本里的每个字转成数字编号，模型只能处理数字，不能直接处理文字。"""
    return [char_to_id.get(char, char_to_id["<UNK>"]) for char in text]


def build_sample(ni_position=None):
    """
    生成一条样本。

    ni_position 是“你”的位置，下标从 0 开始。
    标签也从 0 开始，所以：
        第 1 类 -> 标签 0
        第 2 类 -> 标签 1
        ...
    """

    # 如果“你”的位置没有指定，则随机选择一个位置
    if ni_position is None:
        ni_position = random.randint(0, SEQ_LEN - 1)

    # 从字符池中随机选择 SEQ_LEN -5 个字符
    chars = random.choices(CHAR_POOL, k=SEQ_LEN)

    # 将“你”字放在随机位置
    chars[ni_position] = NI_CHAR
    text = "".join(chars)
    label = ni_position
    return text, label


def build_dataset(sample_num, char_to_id):
    """生成训练集或测试集。"""
    input_ids = []
    labels = []

    for index in range(sample_num):
        # 让 5 个类别尽量平均出现，训练更稳定。
        ni_position = index % CLASS_NUM
        text, label = build_sample(ni_position)
        input_ids.append(encode_text(text, char_to_id))
        labels.append(label)

    return torch.LongTensor(input_ids), torch.LongTensor(labels)


class TextRnnClassifier(nn.Module):
    """RNN / LSTM 文本分类模型。"""

    def __init__(self, vocab_size, model_type="rnn", embedding_dim=16, hidden_size=32):
        super().__init__()

        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.model_type == "rnn":
            self.encoder = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        elif self.model_type == "lstm":
            self.encoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        else:
            raise ValueError("model_type 只能是 'rnn' 或 'lstm'")

        self.classifier = nn.Linear(hidden_size, CLASS_NUM)

    def forward(self, x, y=None):
        # x.shape = [batch_size, 5]
        embedding = self.embedding(x)

        # embedding.shape = [batch_size, 5, embedding_dim]
        # self.encoder是RNN或LSTM模型，返回output和hidden
        # output.shape = [batch_size, 5, hidden_size] --- 因为只需要判断在第几个位置，所以只需要最后一层的隐藏状态。
        # hidden.shape = [num_layers, batch_size, hidden_size]
        _, hidden = self.encoder(embedding)

        if self.model_type == "lstm":
            hidden = hidden[0]

        # 取最后一层的隐藏状态，作为整句话的表示。
        sentence_vector = hidden[-1]

        # 将句子表示映射到类别数量
        logits = self.classifier(sentence_vector)

        # 如果标签存在，计算loss
        if y is not None:
            return F.cross_entropy(logits, y)

        return F.softmax(logits, dim=1)


def evaluate(model, test_x, test_y):
    """计算准确率。"""
    model.eval()
    with torch.no_grad():
        y_pred = model(test_x)
        pred_labels = torch.argmax(y_pred, dim=1)
        correct = (pred_labels == test_y).sum().item()

    return correct / len(test_y)


def train(model_type, char_to_id, epoch_num=20, batch_size=64, train_sample=2000, learning_rate=0.01):
    """训练一个 RNN 或 LSTM 模型。"""
    torch.manual_seed(42)
    random.seed(42)

    train_x, train_y = build_dataset(train_sample, char_to_id)
    test_x, test_y = build_dataset(500, char_to_id)

    model = TextRnnClassifier(len(char_to_id), model_type=model_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\n开始训练 %s 模型" % model_type.upper())

    for epoch in range(epoch_num):
        model.train()
        permutation = torch.randperm(train_sample)
        train_x = train_x[permutation]
        train_y = train_y[permutation]

        loss_list = []
        for batch_index in range(train_sample // batch_size):
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            batch_x = train_x[start_index:end_index]
            batch_y = train_y[start_index:end_index]

            loss = model(batch_x, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_list.append(loss.item())

        acc = evaluate(model, test_x, test_y)
        avg_loss = sum(loss_list) / len(loss_list)
        print("第 %02d 轮，平均 loss: %.4f，测试准确率: %.4f" % (epoch + 1, avg_loss, acc))

    OUTPUT_DIR.mkdir(exist_ok=True)
    model_path = OUTPUT_DIR / ("%s_text_classifier.bin" % model_type)
    torch.save(model.state_dict(), model_path)
    print("%s 模型已保存到: %s" % (model_type.upper(), model_path))
    return model


def predict(model, texts, char_to_id):
    """用训练好的模型预测新文本。"""
    model.eval()

    input_ids = []
    for text in texts:
        if len(text) != SEQ_LEN:
            raise ValueError("输入文本必须是 5 个字: %s" % text)
        input_ids.append(encode_text(text, char_to_id))

    with torch.no_grad():
        result = model(torch.LongTensor(input_ids))

    for text, prob in zip(texts, result):
        pred_label = int(torch.argmax(prob))
        print("输入: %s，预测: 第 %d 类，置信度: %.4f" % (text, pred_label + 1, float(prob.max())))


if __name__ == "__main__":
    char_to_id = build_vocab()

    rnn_model = train("rnn", char_to_id)
    predict(rnn_model, ["你真好呀哈", "我你真好呀", "我们你真好", "我们真你好", "我们真好你"], char_to_id)

    lstm_model = train("lstm", char_to_id)
    predict(lstm_model, ["你真好呀哈", "我你真好呀", "我们你真好", "我们真你好", "我们真好你"], char_to_id)
