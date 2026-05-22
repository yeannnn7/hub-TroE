"""
基于 Transformer 的单向语言模型训练脚本，含 PPL 计算和文本生成。
用法:
    python language_model.py --epochs 20
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 位置编码 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────── Causal Mask ───────────────────────────

def generate_causal_mask(seq_len, device):
    """生成因果注意力掩码（下三角矩阵），防止看到未来信息"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # 转换为 Transformer 需要的格式: (seq_len, seq_len)
    # 0 表示不参与注意力，1 表示参与
    return mask


# ─────────────────────────── Transformer 语言模型 ───────────────────────────

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # 词嵌入 + 位置编码
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        # Transformer Decoder 层（使用官方的 TransformerDecoderLayer）
        # 注意：解码器层自带因果掩码支持
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        batch_size, seq_len = x.shape
        
        # Embedding + 位置编码
        e = self.embed(x) * math.sqrt(self.d_model)  # 缩放嵌入
        e = self.pos_enc(e)
        
        # 生成因果掩码: (seq_len, seq_len)
        causal_mask = generate_causal_mask(seq_len, x.device)
        
        # 转换为 Transformer 需要的格式 (seq_len, seq_len)，False 表示可看到
        # 对于解码器的 tgt_mask，1 表示屏蔽，0 表示允许
        tgt_mask = (causal_mask == 0).float()
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, 0.0)
        
        # Transformer Decoder: 需要 memory 参数，对于单向语言模型，memory 可以是零张量
        memory = torch.zeros(batch_size, 1, self.d_model, device=x.device)
        
        # 通过 Transformer
        out = self.transformer(
            tgt=e,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None
        )
        
        logits = self.fc(self.drop(out))  # (batch, seq_len, vocab_size)
        return logits


# ─────────────────────────── 简化版 Transformer（推荐用于训练） ───────────────────────────

class SimpleTransformerLM(nn.Module):
    """
    简化版 Transformer LM：直接使用 nn.TransformerEncoder 加因果掩码。
    这种方式更简洁，且训练速度更快。
    """
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        
        # Embedding + 位置编码
        e = self.embed(x) * math.sqrt(self.d_model)
        e = self.pos_enc(e)
        
        # 生成因果掩码: (seq_len, seq_len)，True 表示屏蔽
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # 通过 Transformer Encoder（带因果掩码 = 单向语言模型）
        out = self.transformer(e, mask=causal_mask)
        
        logits = self.fc(self.drop(out))
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


# ─────────────────────────── 文本生成 ───────────────────────────

@torch.no_grad()
def generate(model, start_text, char2idx, idx2char, device, max_len=100, temperature=0.8):
    """
    基于给定的开头文本生成后续内容。
    
    参数:
        model: 训练好的模型
        start_text: 开头文本（字符串）
        char2idx: 字符到索引的映射
        idx2char: 索引到字符的映射
        device: 设备
        max_len: 生成的最大长度
        temperature: 温度参数（<1.0 更保守，>1.0 更随机）
    """
    model.eval()
    
    # 将开头文本转为索引
    ids = [char2idx.get(c, 0) for c in start_text]
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)
    
    generated = list(ids)
    
    for _ in range(max_len):
        # 如果输入太长，只保留最后 max_len 个 token
        if input_ids.size(1) > model.pos_enc.pe.size(1):
            input_ids = input_ids[:, -model.pos_enc.pe.size(1):]
        
        # 前向传播
        logits = model(input_ids)  # (1, seq_len, vocab_size)
        
        # 取最后一个时间步的 logits
        next_logits = logits[0, -1, :] / temperature
        
        # Softmax 得到概率分布，并采样
        probs = F.softmax(next_logits, dim=-1)
        
        # Top-p (nucleus) 采样：保留累积概率达到 0.9 的 token
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumsum_probs > 0.9
        sorted_indices_to_remove[0] = False  # 至少保留第一个
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs /= sorted_probs.sum()
        
        # 采样一个 token
        next_token = sorted_indices[torch.multinomial(sorted_probs, 1)]
        
        # 如果生成结束符（可以用某个特殊字符），可以停止
        # 这里简单处理：如果生成 <unk> 或索引 0 就继续
        
        generated.append(next_token.item())
        
        # 更新输入
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    # 将索引转回文本
    output_text = ''.join([idx2char[idx] for idx in generated])
    return output_text


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--seq_len",       type=int,   default=128)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--d_model",       type=int,   default=256)
    parser.add_argument("--nhead",         type=int,   default=4)
    parser.add_argument("--num_layers",    type=int,   default=3)
    parser.add_argument("--dim_feedforward", type=int, default=512)
    parser.add_argument("--dropout",       type=float, default=0.1)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--val_ratio",     type=float, default=0.05)
    parser.add_argument("--corpus",        default="*.txt")
    parser.add_argument("--save",          default="transformer_lm.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: Transformer")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        print("未找到 .txt 文件，自动生成示例文本用于训练...")
        sample_texts = [
            "你好世界这是一个测试语料库用于验证语言模型训练流程是否正常。",
            "这里有一些不同长度的句子用来测试模型的训练和评估功能。",
            "语言模型需要大量文本数据才能学到有意义的语言模式。",
            "深度学习是机器学习的一个分支它使用神经网络来学习数据表示。",
            "自然语言处理是人工智能领域的一个重要方向。",
            "Transformer架构在2017年被提出它彻底改变了NLP领域。",
            "注意力机制允许模型关注输入序列中不同位置的信息。",
        ]
        text = "。".join(sample_texts) * 200  # 增加数据量
    
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\\n".join(lines[:split])
    val_text   = "\\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型（使用简化版 Transformer）
    model = SimpleTransformerLM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_len=args.seq_len,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_val_ppl = float("inf")

    print(f"\\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # ─────────────────────────── 文本生成演示 ───────────────────────────
    print("\\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    # 加载最佳模型
    checkpoint = torch.load(args.save, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # 测试不同的开头
    prompts = [
        "好逛的地方",
        "算法",
        "大模型",
        "今天天气",
        "人工智能",
    ]
    
    for prompt in prompts:
        generated_text = generate(
            model, prompt, char2idx, idx2char, device,
            max_len=50, temperature=0.8
        )
        print(f"\\n开头: {prompt}")
        print(f"生成: {generated_text}")
        print("-" * 40)


if __name__ == "__main__":
    main()
