"""
人民日报 NER 训练脚本

训练 BERT + Linear 或 BERT + CRF，在 validation 上监控 entity F1，
保存 val_f1 最优 checkpoint 到 outputs/checkpoints/。
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from dataset import build_label_schema, build_dataloaders, collect_bio_sequences
from model import build_model
from seqeval.metrics import f1_score as seqeval_f1

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


# 验证集评估：seqeval 在 entity 级别算 F1（与 evaluate.py 一致）
def evaluate_epoch(model: nn.Module, loader: DataLoader, id2label: dict, device: torch.device, use_crf: bool) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    # 推理
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
          
            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()
            total_loss += loss.item()
            labels_np = labels.cpu().tolist()

            # 去掉 -100 位置，得到与 seqeval 兼容的 BIO 字符串序列
            batch_golds, batch_preds = collect_bio_sequences(
                labels_np, pred_ids_list, id2label, use_crf=use_crf
            )
            all_golds.extend(batch_golds)
            all_preds.extend(batch_preds)

        # 计算平均损失和 F1 分数
        avg_loss = total_loss / len(loader)

        # 计算 F1 分数
        entity_f1 = seqeval_f1(all_golds, all_preds)
        print(f"  评估结果：loss={avg_loss:.4f}, f1={entity_f1:.4f}")
       
        return avg_loss, entity_f1


# 训练一个 epoch
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: Optimizer, scheduler: LRScheduler, device: torch.device, epoch: int, total_epochs: int, grad_accum: int) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)

    # 训练
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        # 计算损失
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        # 梯度累积
        (loss / grad_accum).backward()
        total_loss += loss.item()
        # 更新优化器和学习率调度器
        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    # 计算平均损失
        # 梯度累积：最后不足 grad_accum 的 batch 也要 step，否则丢失尾部梯度
        remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    return total_loss / len(loader)

# 主函数
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}\n")
    # 构建标签体系
    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    # 构建数据加载器
    bert_path = Path(args.bert_path).resolve()
    if not bert_path.exists():
        raise FileNotFoundError(f"找不到 BERT 模型目录：{bert_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(bert_path), use_fast=True, local_files_only=True
    )
    train_loader, val_loader, _ = build_dataloaders(tokenizer=tokenizer, label2id=label2id, batch_size=args.batch_size, max_length=args.max_length, data_dir=DATA_DIR)
    # 构建模型
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    # 分层学习率：BERT 用小 lr 微调，分类头/CRF 用更大 lr 加快收敛
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr}, 
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print(f"总训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if args.use_crf else "linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []
    # 训练
    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, epoch, args.epochs, args.grad_accum)
        # 评估一个 epoch
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        # 记录日志
        log_records.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_f1": val_f1,
        })
        # 按 validation entity F1 保存最优 checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": args.use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "label2id": label2id,
                    "id2label": id2label,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ✓ 新最优模型已保存 → {ckpt_path}  (val_f1={val_f1:.4f})")
    # 保存日志
    with open(log_path, "w") as f:
        json.dump(log_records, f, indent=4)
    print(f"\n训练完成！最优 val_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")

# 解析参数
def parse_args():
    parser = argparse.ArgumentParser(description="训练 BERT NER 模型")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()

if __name__ == "__main__":
    main()