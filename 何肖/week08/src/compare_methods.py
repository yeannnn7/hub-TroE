"""
多方法效果对比脚本

如何运行
python compare_methods.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder, plot_similarity_distribution
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "bq_corpus"
BERT_PATH  = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"

# ── 方法定义 ──────────────────────────────────────────────────────────────
METHODS = [
    {
        "key":       "biencoder_cosine",
        "label":     "BiEncoder\n(CosineEmbeddingLoss)",
        "ckpt":      "biencoder_cosine_best.pt",
        "type":      "biencoder",
        "color":     "#2196F3",
    },
    {
        "key":       "biencoder_triplet",
        "label":     "BiEncoder\n(TripletLoss)",
        "ckpt":      "biencoder_triplet_best.pt",
        "type":      "biencoder",
        "color":     "#4CAF50",
    },
    {
        "key":       "crossencoder",
        "label":     "CrossEncoder\n(CrossEntropyLoss)",
        "ckpt":      "crossencoder_best.pt",
        "type":      "crossencoder",
        "color":     "#FF9800",
    },
]

# ── 加载并评估单个方法 ─────────────────────────────────────────────────────
def load_and_eval(method, tokenizer, device, split, batch_size):
    ckpt_path = CKPT_DIR / method["ckpt"]
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    data_path = DATA_DIR / f"{split}.jsonl"

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds     = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)
    else:
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"]  = ckpt
    return metrics

def plot_comparison_bar(results, save_path):
    """绘制方法对比柱状图。"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(results)), [r["accuracy"] for r in results], color=[r["color"] for r in results])
    plt.xlabel("方法")
    plt.ylabel("准确率")
    plt.title("方法对比")
    plt.xticks(range(len(results)), [r["label"] for r in results])
    plt.savefig(save_path)
    plt.close()
    print(f"  图表已保存 → {save_path}")

def plot_confusion_matrix(cross_results, save_path):
    """绘制 CrossEncoder 混淆矩阵。"""
    m = cross_results[0]
    preds = np.argmax(m["logits"], axis=1)
    cm = confusion_matrix(m["labels"], preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["不相似", "相似"])
    ax.set_yticklabels(["不相似", "相似"])
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title("CrossEncoder Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_sim_distributions(biencoder_results, save_path):
    """绘制 BiEncoder 相似度分布图。"""
    plt.figure(figsize=(10, 6))
    plt.hist(biencoder_results[0]["similarities"], bins=50, color="red", alpha=0.5, label="不相似")
    plt.hist(biencoder_results[1]["similarities"], bins=50, color="blue", alpha=0.5, label="相似")
    plt.xlabel("相似度")
    plt.ylabel("数量")
    plt.title("BiEncoder 相似度分布")
    plt.legend()
    plt.savefig(save_path)  
    plt.close()
    print(f"  图表已保存 → {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="多方法效果对比脚本")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  评估集: {args.split}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for method in METHODS:
        print(f"\n{'='*55}")
        print(f"加载 {method['key']} ...")
        metrics = load_and_eval(method, tokenizer, device, args.split, args.batch_size)
        if metrics is None:
            continue
        metrics.update({
            "label": method["label"],
            "color": method["color"],
            "key":   method["key"],
            "type":  method["type"],
        })
        all_results.append(metrics)

    if not all_results:
        print("没有可用的 checkpoint，请先运行训练脚本。")
        return

    bi_results = [m for m in all_results if m["type"] == "biencoder"]
    if len(bi_results) == 2:    
        a, b = bi_results
        delta_acc = b["accuracy"] - a["accuracy"]
        delta_f1 = b["f1"] - a["f1"]
        print(f"\n  Cosine vs Triplet (Δ):")
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")
        if abs(delta_f1) < 0.01:
            print("    → 两种 Loss 差距不大（1 epoch + 少量三元组限制了 Triplet 的优势）")
        elif delta_f1 > 0:
            print("    → TripletLoss 更优，三元组对语义距离的约束更精确")
        else:
            print("    → CosineEmbeddingLoss 更优，AFQMC 数据量下直接对标签优化更稳定")

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):  # numpy scalar / torch scalar
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]
    log_path = LOG_DIR / "method_comparison.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, FIG_DIR / "method_comparison_bar.png")

    bi_with_sim = [m for m in all_results if m["type"] == "biencoder" and "similarities" in m]
    if bi_with_sim:
        plot_sim_distributions(bi_with_sim, FIG_DIR / "biencoder_sim_distributions.png")

    cross_results = [m for m in all_results if m["type"] == "crossencoder"]
    if cross_results:
        plot_confusion_matrix(cross_results, FIG_DIR / "crossencoder_confusion_matrix.png")

if __name__ == "__main__":
    main()