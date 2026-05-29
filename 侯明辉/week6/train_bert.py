"""
BERT Fine-Tuning 文本分类训练

教学重点：
  1. 预训练 + 微调范式：BERT 已学过语言知识，微调只需少量 epoch
  2. 差分学习率：BERT 层用小 lr（2e-5），分类头用 5 倍大 lr（1e-4）
  3. Warmup + Linear Decay 调度：避免训练初期大梯度破坏预训练权重
  4. 判别式分类：直接输出 logits，效率远高于 LLM 的 generate 解码

使用方式：
  python train_bert.py                            # 默认参数
  python train_bert.py --epochs 5 --lr 3e-5       # 自定义训练参数
  python train_bert.py --num_train 2000           # 快速演示
  python train_bert.py --pool mean                # 均值池化

依赖：
  pip install torch transformers scikit-learn tqdm
"""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from shared import (
    DATA_DIR, BERT_PATH, OUTPUT_DIR, NUM_LABELS,
    load_raw_data, evaluate_bert, set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT Fine-Tuning 文本分类训练")
    parser.add_argument("--num_train",   default=5000,  type=int,
                        help="训练样本数，-1 使用全部")
    parser.add_argument("--num_val",     default=1000,  type=int,
                        help="验证样本数")
    parser.add_argument("--batch_size",  default=32,    type=int)
    parser.add_argument("--max_length",  default=64,    type=int,
                        help="序列最大长度")
    parser.add_argument("--epochs",      default=3,     type=int)
    parser.add_argument("--lr",          default=2e-5,  type=float,
                        help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", default=5.0,  type=float,
                        help="分类头学习率倍数")
    parser.add_argument("--pool",        default="cls",
                        choices=["cls", "mean", "max"],
                        help="向量提取策略")
    parser.add_argument("--bert_path",   default=str(BERT_PATH))
    parser.add_argument("--seed",        default=42,    type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ────────────────────────────────────────────────────────────
    train_raw, val_raw = load_raw_data(args.num_train, args.num_val, args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # ── 数据集（将子集写入临时文件供 TNEWSDataset 加载） ───────────────────
    from dataset import TNEWSDataset
    import transformers

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        with open(tmp_dir / "train.json", "w", encoding="utf-8") as f:
            json.dump(train_raw, f, ensure_ascii=False)
        with open(tmp_dir / "val.json", "w", encoding="utf-8") as f:
            json.dump(val_raw, f, ensure_ascii=False)
        shutil.copy(DATA_DIR / "label_map.json", tmp_dir / "label_map.json")

        train_ds = TNEWSDataset(tmp_dir / "train.json", tokenizer, args.max_length)
        val_ds   = TNEWSDataset(tmp_dir / "val.json",   tokenizer, args.max_length)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── 模型 ────────────────────────────────────────────────────────────────
    from model import BertClassifier

    _prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    model = BertClassifier(args.bert_path, num_labels=NUM_LABELS, pool=args.pool)
    transformers.logging.set_verbosity(_prev)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    n_bert   = sum(p.numel() for p in model.bert.parameters()) / 1e6
    print(f"模型参数量: {n_params:.1f}M (BERT: {n_bert:.1f}M, 分类头: "
          f"{(n_params - n_bert)*1000:.1f}K)")
    print(f"池化策略: {args.pool}")

    # ── 差分学习率 ──────────────────────────────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"总训练步数: {total_steps}, warmup: {warmup_steps}")

    criterion = nn.CrossEntropyLoss()

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
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss   = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            preds = logits.argmax(dim=-1)
            total_loss    += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            pbar.set_postfix(loss=f"{total_loss/total_samples:.4f}",
                             acc=f"{total_correct/total_samples:.4f}")

        train_loss = total_loss / total_samples
        train_acc  = total_correct / total_samples

        # --- Eval ---
        val_acc, val_f1 = evaluate_bert(model, val_loader, device)

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
        "method": f"BERT (bert-base-chinese, {args.pool} pooling)",
        "params_M": n_params,
        "best_val_acc": best_val_acc,
        "epochs": args.epochs,
        "pool": args.pool,
        "log": log_records,
    }
    with open(OUTPUT_DIR / "bert_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}")
    print(f"结果已保存 → {OUTPUT_DIR / 'bert_result.json'}")


if __name__ == "__main__":
    main()
