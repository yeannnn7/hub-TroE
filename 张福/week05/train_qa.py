"""
问答模型训练脚本

使用Transformer模型训练一个微型问答系统。
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import random
import numpy as np

from transformer import Transformer
from corpus import get_question_answer_pairs


class Vocabulary:
    """词汇表类，用于将字符/词转换为索引"""
    
    def __init__(self):
        self.char2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3, "<SEP>": 4}
        self.idx2char = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>", 4: "<SEP>"}
        self.n_chars = 5
    
    def build_vocab(self, texts):
        """从文本列表构建词汇表"""
        for text in texts:
            for char in text:
                if char not in self.char2idx:
                    self.char2idx[char] = self.n_chars
                    self.idx2char[self.n_chars] = char
                    self.n_chars += 1
    
    def encode(self, text, add_sos=False, add_eos=False):
        """将文本转换为索引列表"""
        indices = []
        if add_sos:
            indices.append(self.char2idx["<SOS>"])
        for char in text:
            indices.append(self.char2idx.get(char, self.char2idx["<UNK>"]))
        if add_eos:
            indices.append(self.char2idx["<EOS>"])
        return indices
    
    def decode(self, indices):
        """将索引列表转换为文本"""
        chars = []
        for idx in indices:
            if idx == self.char2idx["<EOS>"]:
                break
            if idx not in [self.char2idx["<PAD>"], 
                          self.char2idx["<SOS>"],
                          self.char2idx["<EOS>"]]:
                chars.append(self.idx2char.get(idx, "<UNK>"))
        return "".join(chars)


class QADataset(Dataset):
    """问答数据集"""
    
    def __init__(self, pairs, vocab, max_len=512):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        question, answer = self.pairs[idx]
        # 将语料 转为 字典映射索引
        question_enc = self.vocab.encode(question)
        answer_enc = self.vocab.encode(answer, add_eos=True)
        
        # 在问题和答案之间添加分隔符
        sep_enc = [self.vocab.char2idx["<SEP>"]]
        combined_enc = question_enc + sep_enc + answer_enc
        
        if len(combined_enc) > self.max_len:
            combined_enc = combined_enc[:self.max_len]

        # print(combined_enc,type(combined_enc))

        #将字符串，问答 ，转化为向量
        combined_tensor = torch.tensor(combined_enc, dtype=torch.long)
        
        return combined_tensor, len(question_enc)


def pad_collate_fn(batch):
    """批处理填充函数"""
    sequences, q_lens = zip(*batch)
    
    q_lens = torch.tensor(q_lens)
    
    padded = torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    
    return padded, q_lens


def train_model(model, train_loader, test_loader, device, num_epochs=50, lr=0.001, warmup_steps=100):
    """SFT监督微调训练

    SFT核心特点：
    1. Teacher Forcing: 训练时使用真实的上一个token作为输入
    2. Shifted Right: 目标序列相对于输入右移一位
    3. 因果Mask: 确保每个位置只能看到当前位置及之前的token
    4. 只计算答案部分的损失: 问题部分不参与损失计算
    """

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.95))

    best_loss = float('inf')
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, (seq, q_lens) in enumerate(train_loader):
            seq = seq.to(device)
            q_lens = q_lens.to(device)

            batch_size, seq_len = seq.size()

            optimizer.zero_grad()

            input_seq = seq[:, :-1] # 所有行   到  从开始-倒数第二列的向量
            target_seq = seq[:, 1:] # 所有行   到  从第二个到 最后的列向量 向后多取一列

            output = model.forward(input_seq)
            # print("ourput向量维度:",output.shape)

            loss = 0.0
            correct = 0
            tokens = 0

            for i in range(batch_size):
                start_idx = q_lens[i] + 1  # +1 跳过分隔符
                end_idx = seq_len - 1

                if start_idx < end_idx:
                    output_slice = output[i, start_idx:end_idx, :] # 向量
                    target_slice = target_seq[i, start_idx:end_idx] # 获取行数
                    # print("output_slice维度",output_slice.shape) #torch.Size([74, 566])
                    # print("target_slice维度", target_slice.shape) #torch.Size([74])

                    loss += criterion(output_slice, target_slice)

                    preds = torch.argmax(output_slice, dim=-1)
                    correct += (preds == target_slice).sum().item()
                    tokens += target_slice.size(0)

            loss = loss / batch_size if batch_size > 0 else 0
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            global_step += 1

            total_loss += loss.item()
            total_correct += correct
            total_tokens += tokens

        avg_train_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        model.eval()
        total_test_loss = 0
        total_test_correct = 0
        total_test_tokens = 0

        with torch.no_grad():
            for seq, q_lens in test_loader:
                seq = seq.to(device)
                q_lens = q_lens.to(device)

                input_seq = seq[:, :-1]
                target_seq = seq[:, 1:]

                output = model(input_seq)

                batch_size, seq_len = target_seq.size()
                batch_loss = 0.0
                correct = 0
                tokens = 0

                for i in range(batch_size):
                    start_idx = q_lens[i] + 1  # +1 跳过分隔符
                    end_idx = seq_len - 1

                    if start_idx < end_idx:
                        output_slice = output[i, start_idx:end_idx, :] # 第i个样本,第 start_idx ~  end_idx-1行  ，所有列  切片
                        target_slice = target_seq[i, start_idx:end_idx] # 第 i 行，start_idx ~  end_idx-1列

                        batch_loss += criterion(output_slice, target_slice)

                        preds = torch.argmax(output_slice, dim=-1)
                        correct += (preds == target_slice).sum().item()
                        tokens += target_slice.size(0)

                total_test_loss += (batch_loss / batch_size if batch_size > 0 else 0).item()
                total_test_correct += correct
                total_test_tokens += tokens

        avg_test_loss = total_test_loss / len(test_loader)
        test_accuracy = total_test_correct / total_test_tokens if total_test_tokens > 0 else 0

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {accuracy:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {test_accuracy:.4f}")

        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save(model.state_dict(), "ask_answer.pt")
            print(f"  -> 保存最佳模型 (loss: {best_loss:.4f})")

    return model


def create_split_dataset(corpus_data, train_ratio=0.8, random_seed=42):
    """划分训练集和测试集"""
    random.seed(random_seed)
    random.shuffle(corpus_data)
    
    split_idx = int(len(corpus_data) * train_ratio)
    train_data = corpus_data[:split_idx]
    test_data = corpus_data[split_idx:]
    
    print(f"数据集划分: 训练集 {len(train_data)} 条, 测试集 {len(test_data)} 条")
    
    return train_data, test_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    pairs = get_question_answer_pairs()

    questions = [p[0] for p in pairs]
    answers = [p[1] for p in pairs]

    vocab = Vocabulary()
    vocab.build_vocab(questions + answers)
    print(f"词汇表大小: {vocab.n_chars}")

    train_data, test_data = create_split_dataset(pairs)

    max_len = 512
    train_dataset = QADataset(train_data, vocab, max_len)
    test_dataset = QADataset(test_data, vocab, max_len)
    
    batch_size = 8
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn
    )
    
    vocab_size = vocab.n_chars
    embed_dim = 256
    num_heads = 8
    ff_dim = 1024
    num_layers = 4
    max_len = 100
    dropout = 0.1
    
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    )
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n开始训练...")
    model = train_model(model, train_loader, test_loader, device, num_epochs=20, lr=5e-4)
    
    print("\n训练完成！模型已保存为 ask_answer.pt")
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": vocab,
        "vocab_size": vocab_size,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "num_layers": num_layers,
        "max_len": max_len
    }, "ask_answer.pt")
    print("完整模型信息已保存!")


if __name__ == "__main__":
    main()