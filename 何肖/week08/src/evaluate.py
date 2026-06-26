"""
文本匹配评估工具（可作模块导入，也可独立运行）

如何使用：
# BiEncoder - cosine
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt
  --split test

# BiEncoder - triplet
python evaluate.py --model_type biencoder \
  --ckpt ../outputs/checkpoints/biencoder_triplet_best.pt \
  --split test

# CrossEncoder
python evaluate.py --model_type crossencoder \
  --ckpt ../outputs/checkpoints/crossencoder_best.pt
  --split test
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
FIG_DIR = ROOT / "outputs" / "figures"


def _find_best_threshold(sims, labels):
    """枚举 [0.0, 1.0] 区间 101 个候选阈值，返回使 weighted-F1 最高的那个。"""
    best_f1, best_thresh = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 101):
        preds = (sims >= t).astype(int)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return float(best_thresh)

# ── 评估 BiEncoder ──────────────────────────────────────────────────────────
@torch.no_grad()
def eval_biencoder(model, loader, device, find_threshold=True, threshold=0.5):
    """
    BiEncoder 评估：计算每对句子的余弦相似度，然后在 val 集上搜索最优阈值。

    """             
    model.eval()
    all_sims, all_labels = [], []

    for batch in loader:
        batch_a = {
            "input_ids":      batch["input_ids_a"].to(device),
            "attention_mask": batch["attention_mask_a"].to(device),
            "token_type_ids": batch["token_type_ids_a"].to(device),
        }
        batch_b = {
            "input_ids":      batch["input_ids_b"].to(device),
            "attention_mask": batch["attention_mask_b"].to(device),
            "token_type_ids": batch["token_type_ids_b"].to(device),
        }
        emb_a, emb_b = model(batch_a, batch_b)
        sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().tolist()
        all_sims.extend(sims)
        all_labels.extend(batch["label"].tolist())

    sims   = np.array(all_sims)
    labels = np.array(all_labels)
    
    if find_threshold:
        threshold = _find_best_threshold(sims, labels)

    preds    = (sims >= threshold).astype(int)
    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, average="weighted", zero_division=0)

    # AUC：若 labels 只有一类（如 AFQMC test 全为 0），跳过
    try:
        auc = roc_auc_score(labels, sims)
    except ValueError:
        auc = float("nan")

    return {
        "similarities": all_sims,
        "labels":       all_labels,
        "accuracy":     accuracy,
        "f1":           f1,
        "threshold":    threshold,
        "auc":          auc,
    }


# ── 评估 CrossEncoder ─────────────────────────────────────────────────────
@torch.no_grad()
def eval_crossencoder(model, loader, device):
    """
    CrossEncoder 评估：argmax 得预测标签，与二分类任务一致。
    """
    model.eval()
    all_logits, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels         = batch["label"]

        logits = model(input_ids, attention_mask, token_type_ids).cpu()
        all_logits.extend(logits.tolist())
        all_labels.extend(labels.tolist())

    logits_np = np.array(all_logits)
    preds     = np.argmax(logits_np, axis=1)
    labels    = np.array(all_labels)
    accuracy  = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, average="weighted", zero_division=0)

    try:
        probs = torch.softmax(torch.tensor(logits_np), dim=-1)[:, 1].numpy()
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {
        "logits":   all_logits,
        "labels":   all_labels,
        "accuracy": accuracy,
        "f1":       f1,
        "auc":      auc,
    }


def plot_similarity_distribution(sims, labels, threshold, save_path, title="相似度分布"):
    """绘制相似度分布图，显示正负样本的分离程度。"""
    plt.figure(figsize=(10, 6))
    plt.hist(sims[labels == 0], bins=50, color="red", alpha=0.5, label="不相似")
    plt.hist(sims[labels == 1], bins=50, color="blue", alpha=0.5, label="相似")
    plt.axvline(x=threshold, color="green", linestyle="--", label=f"最优阈值: {threshold:.2f}")
    plt.xlabel("相似度")
    plt.ylabel("数量")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存 → {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="文本匹配评估工具")
    parser.add_argument("--model_type", type=str, choices=["biencoder", "crossencoder"], required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "bq_corpus"))
    parser.add_argument("--bert_path", type=str, default=str(BERT_PATH))
    parser.add_argument("--split", type=str, default="validation",
                        choices=["train", "validation", "test"])
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--find_threshold", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"加载 checkpoint: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    print(f"训练信息: {ckpt.get('args', {})}")

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    data_path = Path(args.data_dir) / f"{args.split}.jsonl"

    if args.model_type == "biencoder":
        from model import build_biencoder
        from dataset import PairDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_biencoder(
            bert_path=args.bert_path,
            pool=saved_args.get("pool", "mean"),
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds     = PairDataset(data_path, tokenizer, args.max_length)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        metrics = eval_biencoder(model, loader, device)
        print(f"\n{'='*50}")
        print(f"BiEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  最优阈值: {metrics['threshold']:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  AUC     : {metrics['auc']:.4f}")
        
        plot_similarity_distribution(
            metrics["similarities"], metrics["labels"], metrics["threshold"],
            save_path=FIG_DIR / f"biencoder_{args.split}_sim_dist.png",
            title=f"BiEncoder 相似度分布（{args.split}）",
        )

        # 打印分类报告
        preds = (np.array(metrics["similarities"]) >= metrics["threshold"]).astype(int)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")

    elif args.model_type == "crossencoder":     
        from model import build_crossencoder
        from dataset import CrossEncoderDataset
        from torch.utils.data import DataLoader

        saved_args = ckpt.get("args", {})
        model = build_crossencoder(
            bert_path=args.bert_path,
            num_hidden_layers=saved_args.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])

        ds = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        metrics = eval_crossencoder(model, loader, device)
        print(f"\n{'='*50}")
        print(f"CrossEncoder 评估结果（{args.split}，{len(ds)} 条）")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1      : {metrics['f1']:.4f}")
        print(f"  AUC     : {metrics['auc']:.4f}")
        
        preds = np.argmax(metrics["logits"], axis=1)
        print(f"\n{classification_report(metrics['labels'], preds, target_names=['不相似', '相似'])}")

    else:
        raise ValueError(f"未知模型类型: {args.model_type}")

if __name__ == "__main__":
    main()