"""
LSTM 文本分类训练 (BiLSTM)

教学重点：
  1. 字符级词表构建：中文按单字切分，无需分词器
  2. BiLSTM 结构：双向捕获上下文，取最后层隐状态做分类
  3. 从零训练 vs 预训练微调：LSTM 所有参数从随机初始化开始学习
  4. 对比 BERT：参数少（~5M vs ~110M），但效果上限低

使用方式：
  python train_lstm.py                            # 默认参数
  python train_lstm.py --epochs 5 --lr 5e-4       # 自定义训练参数
  python train_lstm.py --num_train 2000           # 快速演示
  python train_lstm.py --hidden_dim 128 --embed_dim 200  # 调整模型结构

依赖：
  pip install torch scikit-learn tqdm
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from shared import (
    DATA_DIR, OUTPUT_DIR, NUM_LABELS,
    load_raw_data, evaluate_discriminative, set_seed,
)


# ══════════════════════════════════════════════════════════════════════════════
# 字符级词表
# ══════════════════════════════════════════════════════════════════════════════

class Vocab:
    """字符级词表构建器，将中文字符映射为整数 ID。"""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self, texts, min_freq=2):
        # 统计字符频次
        char_counts = {}
        for text in texts:
            for ch in text:
                char_counts[ch] = char_counts.get(ch, 0) + 1

        # 构建词表：特殊 token + 高频字符
        self.char2id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        idx = 2
        for ch, cnt in sorted(char_counts.items()):
            if cnt >= min_freq:
                self.char2id[ch] = idx
                idx += 1

        self.id2char = {v: k for k, v in self.char2id.items()}
        self.pad_id = 0
        self.vocab_size = len(self.char2id)
        print(f"  Vocab: {self.vocab_size} 个字符（min_freq={min_freq}）")

    def encode(self, text, max_length):
        """将文本编码为 ID 序列，截断 + 填充至 max_length。"""
        ids = [self.char2id.get(ch, self.char2id[self.UNK_TOKEN]) for ch in text]
        ids = ids[:max_length]
        ids += [self.pad_id] * (max_length - len(ids))
        return ids


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDataset(Dataset):
    """LSTM 用的 Dataset：字符级编码，返回 input_ids + label。"""

    def __init__(self, data, vocab, max_length=64):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = self.vocab.encode(item["sentence"], self.max_length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label":     torch.tensor(item["label"], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# BiLSTM 分类模型
# ══════════════════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    """
    BiLSTM 文本分类模型

    结构：Embedding → BiLSTM → 取最后时刻隐状态 → Dropout → Linear → logits

    教学对比 BERT：
      - LSTM 需要从零训练 Embedding 和 LSTM 权重
      - BERT 有预训练权重，微调只需少量 epoch
      - LSTM 参数量远小于 BERT（约 5M vs 110M），但效果上限低
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_labels: int = 15,
        dropout: float = 0.3,
        pad_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 双向 LSTM 输出维度 = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, L]
        返回 logits: [B, num_labels]
        """
        # Embedding: [B, L] → [B, L, embed_dim]
        embeds = self.embedding(input_ids)

        # BiLSTM: [B, L, embed_dim] → output [B, L, hidden*2], (h_n, c_n)
        output, (h_n, c_n) = self.lstm(embeds)

        # 取双向最后时刻的隐状态拼接：h_n 形状 [num_layers*2, B, hidden]
        # 取最后一层的正向和反向
        h_forward  = h_n[-2]   # [B, hidden]  最后一层正向
        h_backward = h_n[-1]   # [B, hidden]  最后一层反向
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # [B, hidden*2]

        h_cat = self.dropout(h_cat)
        logits = self.classifier(h_cat)  # [B, num_labels]
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# 训练入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LSTM (BiLSTM) 文本分类训练")
    parser.add_argument("--num_train",   default=5000,  type=int,
                        help="训练样本数，-1 使用全部")
    parser.add_argument("--num_val",     default=1000,  type=int,
                        help="验证样本数")
    parser.add_argument("--batch_size",  default=32,    type=int)
    parser.add_argument("--max_length",  default=64,    type=int,
                        help="序列最大长度")
    parser.add_argument("--epochs",      default=10,    type=int)
    parser.add_argument("--lr",          default=1e-3,  type=float,
                        help="学习率")
    parser.add_argument("--embed_dim",   default=300,   type=int,
                        help="词嵌入维度")
    parser.add_argument("--hidden_dim",  default=256,   type=int,
                        help="LSTM 隐藏层维度")
    parser.add_argument("--num_layers",  default=2,     type=int,
                        help="LSTM 层数")
    parser.add_argument("--dropout",     default=0.3,   type=float)
    parser.add_argument("--seed",        default=42,    type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ────────────────────────────────────────────────────────────
    train_raw, val_raw = load_raw_data(args.num_train, args.num_val, args.seed)

    # ── 构建词表 ────────────────────────────────────────────────────────────
    texts = [item["sentence"] for item in train_raw]
    vocab = Vocab(texts, min_freq=2)

    # ── 构建数据集 ──────────────────────────────────────────────────────────
    train_ds = LSTMDataset(train_raw, vocab, args.max_length)
    val_ds   = LSTMDataset(val_raw,   vocab, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── 构建模型 ────────────────────────────────────────────────────────────
    model = LSTMClassifier(
        vocab_size=vocab.vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_labels=NUM_LABELS,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"模型参数量: {n_params:.2f}M")
    print(f"  Embedding: {args.embed_dim}d, LSTM hidden: {args.hidden_dim}, "
          f"layers: {args.num_layers}, bidirectional=True")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── 训练循环 ────────────────────────────────────────────────────────────
    log_records = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]",
                     leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["label"].to(device)

            logits = model(input_ids)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            preds = logits.argmax(dim=-1)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                             acc=f"{total_correct/total_samples:.4f}")

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # --- Eval ---
        val_acc, val_f1 = evaluate_discriminative(model, val_loader, device)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f} | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1, "elapsed_s": elapsed,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # ── 保存结果 ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "LSTM (BiLSTM)",
        "params_M": n_params,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "log": log_records,
    }
    with open(OUTPUT_DIR / "lstm_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"结果已保存 → {OUTPUT_DIR / 'lstm_result.json'}")


if __name__ == "__main__":
    main()
