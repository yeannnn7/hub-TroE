"""
CrossEncoder 池化对比训练脚本

目标：
  1. 支持 cls / mean / max 三种池化方式
  2. 每隔固定步数输出 train loss、val loss、val f1 的变化
  3. 保存按 pool 区分的 checkpoint 与训练日志

使用方式：
  python train_crossencoder_pooling.py --pool cls
  python train_crossencoder_pooling.py --pool mean --eval_steps 500 --epochs 3
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from transformers import BertTokenizer, get_linear_schedule_with_warmup

from dataset import build_crossencoder_loaders
from evaluate import eval_crossencoder_with_loss
from model import build_crossencoder


def get_default_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")
    return torch.device("cpu")


ROOT = Path(__file__).parent.parent
MODEL_ROOT = Path("/root/rpc_stu/model")
DATA_DIR = ROOT / "data" / "bq_corpus"
BERT_PATH = MODEL_ROOT / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"


def evaluate_and_log(model, val_loader, criterion, device, global_step, train_loss, log_records):
    val_metrics, val_loss = eval_crossencoder_with_loss(model, val_loader, device, criterion)
    val_acc = val_metrics["accuracy"]
    val_f1 = val_metrics["f1"]
    print(
        f"step={global_step} | train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}"
    )
    log_records.append({
        "step": global_step,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_f1": val_f1,
    })
    return val_metrics, val_loss


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, device, args):
    best_val_f1 = 0.0
    global_step = 0
    log_records = []
    ckpt_path = CKPT_DIR / f"crossencoder_{args.pool}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss, running_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [{args.pool}]", leave=False)

        for step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            (loss / args.grad_accum).backward()
            if step % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item() * labels.size(0)
            running_samples += labels.size(0)
            avg_train_loss = running_loss / running_samples
            pbar.set_postfix(loss=f"{avg_train_loss:.4f}", step=global_step)

            if global_step > 0 and global_step % args.eval_steps == 0:
                val_metrics, val_loss = evaluate_and_log(
                    model, val_loader, criterion, device, global_step, avg_train_loss, log_records
                )
                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    torch.save({
                        "epoch": epoch,
                        "step": global_step,
                        "state_dict": model.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "args": vars(args),
                    }, ckpt_path)
                    print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={best_val_f1:.4f})")
                model.train()

        if running_samples > 0 and (not log_records or log_records[-1]["step"] != global_step):
            val_metrics, val_loss = evaluate_and_log(
                model, val_loader, criterion, device, global_step,
                running_loss / running_samples, log_records
            )
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                torch.save({
                    "epoch": epoch,
                    "step": global_step,
                    "state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                    "args": vars(args),
                }, ckpt_path)
                print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={best_val_f1:.4f})")

    return best_val_f1, log_records, ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="CrossEncoder 池化对比训练")
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--pool", default="cls", choices=["cls", "mean", "max"])
    parser.add_argument("--num_hidden_layers", default=12, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = get_default_device()
    print(f"设备: {device}")
    print(
        f"池化策略: {args.pool}  BERT 层数: {args.num_hidden_layers}  "
        f"Epochs: {args.epochs}  eval_steps: {args.eval_steps}"
    )

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    print("\nDataLoader 构建中...")
    train_loader, val_loader, _ = build_crossencoder_loaders(
        args.data_dir, tokenizer,
        max_length=args.max_length, batch_size=args.batch_size,
    )

    print("\n构建模型...")
    model = build_crossencoder(
        bert_path=args.bert_path,
        pool=args.pool,
        num_hidden_layers=args.num_hidden_layers,
    ).to(device)

    bert_params = list(model.bert.parameters())
    head_params = list(model.dropout.parameters()) + list(model.classifier.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = max(1, len(train_loader) * args.epochs // args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    print(f"总训练步数: {total_steps}  Warmup 步数: {warmup_steps}")

    start_time = time.time()
    best_val_f1, log_records, ckpt_path = train(
        model, train_loader, val_loader, optimizer, scheduler, criterion, device, args
    )
    elapsed = time.time() - start_time

    log_path = LOG_DIR / f"crossencoder_{args.pool}_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_f1={best_val_f1:.4f}")
    print(f"总耗时: {elapsed:.0f}s")
    print(f"训练日志 → {log_path}")
    print(f"最优 checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
