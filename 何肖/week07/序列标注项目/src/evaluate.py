"""
人民日报 NER 评估脚本

使用方式：
  python evaluate.py                  # 评估 BERT+Linear（validation）
  python evaluate.py --use_crf        # 评估 BERT+CRF
  python evaluate.py --split test     # 在测试集上评估（peoples_daily 含标签）
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer
from seqeval.metrics import (
    classification_report as seqeval_report,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import build_label_schema, build_dataloaders, collect_bio_sequences
from model import build_model

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
DATA_DIR = ROOT / "data" / "peoples_daily"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计 Linear 头常见的非法 BIO（CRF 解码后通常更少）。

    规则：序列不能以 I-X 开头；I-Y 只能接在 O / B-Y / I-Y 之后。
    """
    stats = {
        "illegal_start": 0,
        "illegal_transition": 0,
        "total_seqs": len(pred_seqs),
    }
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if not curr.startswith("I-"):
                continue
            curr_type = curr[2:]
            if prev == "O":
                stats["illegal_transition"] += 1
            elif prev.startswith("B-") or prev.startswith("I-"):
                if prev[2:] != curr_type:
                    stats["illegal_transition"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """推理并返回 (all_preds, all_golds)。

    gold/pred 等长，均只含「有标签的字」对应 token，不含 [CLS]/[SEP]/子词续字。
    """
    model.eval()
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()
            batch_golds, batch_preds = collect_bio_sequences(
                labels_list, pred_ids_list, id2label, use_crf=use_crf
            )
            all_golds.extend(batch_golds)
            all_preds.extend(batch_preds)

    return all_preds, all_golds


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt
    return {"state_dict": ckpt}


def parse_args():
    parser = argparse.ArgumentParser(description="加载 checkpoint 并评估")
    parser.add_argument("--use_crf", action="store_true")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}\n")

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（O + {num_labels - 1} 个实体标签）")

    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    if not ckpt_path.exists():
        print(f"找不到 checkpoint：{ckpt_path}")
        print(f"请先运行：python train.py {'--use_crf' if args.use_crf else ''}")
        return

    ckpt = _load_checkpoint(ckpt_path, device)
    ckpt_args = ckpt.get("args", {})
    max_length = ckpt_args.get("max_length", args.max_length)

    bert_path = Path(args.bert_path).resolve()
    if not bert_path.exists():
        print(f"找不到 BERT 模型目录：{bert_path}")
        return

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(bert_path),
        num_labels=num_labels,
        dropout=ckpt_args.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])

    if "val_entity_f1" in ckpt:
        print(
            f"加载 checkpoint（epoch={ckpt.get('epoch', '?')}，"
            f"val_f1={ckpt['val_entity_f1']:.4f}）"
        )
    else:
        print(f"加载 checkpoint → {ckpt_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(bert_path), use_fast=True, local_files_only=True
    )
    _, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=max_length,
        data_dir=DATA_DIR,
    )
    loader = val_loader if args.split == "validation" else test_loader
    split_name = args.split

    print(f"\n正在在 [{split_name}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, args.use_crf)

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = f1_score(all_golds, all_preds)

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    illegal_stats = count_illegal_sequences(all_preds)
    print("\n【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  → CRF Viterbi 解码：非法序列 0 条 ✓")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练（3+ epochs）后可降至 0")

    # 保存结果 JSON
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    out_path = LOG_DIR / f"eval_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
