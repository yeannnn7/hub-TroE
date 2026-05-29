"""
训练基于transformer的单向语言模型，并完成文本生成。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
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


# ─────────────────────────── 模型 ───────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=2, num_layers=1, dim_feedforward=256, dropout=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        mask = nn.Transformer.generate_square_subsequent_mask(max_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        # x: (B, T)
        T = x.size(1)
        e = self.embed(x) * math.sqrt(self.d_model)
        e = self.pos_encoder(e)
        e = self.drop(e)
        out = self.transformer(e, mask=self.causal_mask[:T, :T], is_causal=False)
        logits = self.fc(self.drop(out))  # (B, T, V)
        return logits


class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, model_type, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        rnn_cls = nn.LSTM if model_type == "lstm" else nn.RNN
        self.rnn = rnn_cls(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        e = self.drop(self.embed(x))
        out, _ = self.rnn(e)
        logits = self.fc(self.drop(out))   # (B, T, V)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, scaler=None):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0
    use_amp = scaler is not None

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    if total_tokens == 0:
        return float("inf"), float("inf")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


@torch.no_grad()
def generate(model, start_str, char2idx, idx2char, max_len, device, temperature=1.0):
    model.eval()
    ids = [char2idx.get(c) for c in start_str if c in char2idx]
    if not ids:
        return ""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    result = list(start_str)

    for _ in range(max_len):
        # 截断过长的序列，避免显存/内存溢出
        x_in = x if x.size(1) <= 512 else x[:, -512:]
        logits = model(x_in)               # (1, T, V)
        last_logits = logits[0, -1, :] / temperature  # (V,)
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        if next_id not in idx2char:
            break
        result.append(idx2char[next_id])
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)

    return "".join(result)


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=64)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--num_layers", type=int,   default=1)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--nhead",      type=int,   default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_model.pt")
    parser.add_argument("--gen_len",    type=int,   default=200)
    parser.add_argument("--no_amp",     action="store_true", help="禁用混合精度训练")
    parser.add_argument("--no_compile", action="store_true", help="禁用 torch.compile 加速")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    if len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        val_loader = None

    # 模型
    if args.model == "transformer":
        model = TransformerLM(
            vocab_size=vocab_size,
            d_model=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
    else:
        model = LM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model)
        print("已启用 torch.compile 加速")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("已启用 AMP 混合精度训练")

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True, scaler=scaler)
        if val_loader is not None:
            with torch.no_grad():
                va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False, scaler=scaler)
        else:
            va_loss, va_ppl = float("inf"), float("inf")
        scheduler.step()

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

    if best_val_ppl == float("inf"):
        torch.save({
            "model_state": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "args": vars(args),
        }, args.save)

    print(f"\n训练完成。{'最佳验证 PPL: {:.2f}'.format(best_val_ppl) if best_val_ppl != float('inf') else '（无验证集）'}  已保存至 {args.save}")

    # ── 加载最佳模型并生成示例文本 ──
    ckpt = torch.load(args.save, map_location=device, weights_only=False)
    if args.model == "transformer":
        best_model = TransformerLM(
            vocab_size=vocab_size,
            d_model=args.embed_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
        ).to(device)
    else:
        best_model = LM(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            model_type=args.model,
            dropout=args.dropout,
        ).to(device)
    best_model.load_state_dict(ckpt["model_state"])
    best_model.eval()

    # 从训练文本中随机选一个片段作为 prompt
    start_pos = random.randint(0, max(0, len(train_text) - 10))
    prompt = train_text[start_pos: start_pos + 10].replace("\n", "")
    if not prompt:
        prompt = "今天"

    print(f"\nPrompt: {prompt}")
    generated = generate(best_model, prompt, char2idx, idx2char, max_len=args.gen_len, device=device, temperature=0.8)
    print(f"生成文本:\n{generated}")


if __name__ == "__main__":
    main()
