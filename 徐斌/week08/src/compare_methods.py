"""
多方法 / 多数据集效果对比脚本

对比三种文本匹配方式在不同数据集 validation 集上的效果：
  1. BiEncoder + CosineEmbeddingLoss
  2. BiEncoder + TripletLoss
  3. CrossEncoder + CrossEntropyLoss

使用方式：
  # 对比 bq_corpus 与 lcqmc（需先在各数据集上完成训练）
  python compare_methods.py --datasets bq_corpus lcqmc

  # 单数据集
  python compare_methods.py --datasets bq_corpus --split validation

依赖：
  pip install torch transformers scikit-learn matplotlib
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
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataset import PairDataset, CrossEncoderDataset
from evaluate import eval_biencoder, eval_crossencoder
from model import build_biencoder, build_crossencoder

# ── 默认路径 ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
DATA_ROOT  = ROOT / "data"
BERT_PATH  = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
BERT_PATH  = "/Users/xubin/Documents/www/py/llm/model/bert-base-chinese/models--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"

METHODS = [
    {
        "key":   "biencoder_cosine",
        "label": "BiEncoder (Cosine)",
        "ckpt":  "biencoder_cosine_{dataset}_best.pt",
        "type":  "biencoder",
        "color": "#2196F3",
    },
    {
        "key":   "biencoder_triplet",
        "label": "BiEncoder (Triplet)",
        "ckpt":  "biencoder_triplet_{dataset}_best.pt",
        "type":  "biencoder",
        "color": "#4CAF50",
    },
    {
        "key":   "crossencoder",
        "label": "CrossEncoder",
        "ckpt":  "crossencoder_{dataset}_best.pt",
        "type":  "crossencoder",
        "color": "#FF9800",
    },
]

LEGACY_CKPTS = {
    "biencoder_cosine":  "biencoder_cosine_best.pt",
    "biencoder_triplet": "biencoder_triplet_best.pt",
    "crossencoder":      "crossencoder_best.pt",
}


def resolve_ckpt_path(method, dataset):
    """优先使用数据集专属 checkpoint，兼容旧命名。"""
    primary = CKPT_DIR / method["ckpt"].format(dataset=dataset)
    if primary.exists():
        return primary
    legacy = CKPT_DIR / LEGACY_CKPTS[method["key"]]
    if legacy.exists():
        return legacy
    return primary


def load_and_eval(method, tokenizer, device, data_dir, split, batch_size):
    dataset_name = Path(data_dir).name
    ckpt_path = resolve_ckpt_path(method, dataset_name)
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})
    bert_path = saved.get("bert_path", str(BERT_PATH))

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=bert_path,
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    else:
        model = build_crossencoder(
            bert_path=bert_path,
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds = CrossEncoderDataset(
            data_path, tokenizer, max_length=saved.get("max_length", 128),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["ckpt"] = ckpt
    return metrics


def print_dataset_table(dataset_name, results):
    print(f"\n{'='*70}")
    print(f"数据集: {dataset_name}")
    print(f"{'方法':<22} {'Accuracy':>9} {'F1':>9} {'AUC/备注':>12}")
    print(f"{'-'*70}")
    for m in results:
        extra = (f"thr={m['threshold']:.2f}"
                 if m["type"] == "biencoder"
                 else "argmax")
        auc = m.get("auc")
        note = f"auc={auc:.3f}" if auc is not None else extra
        print(f"  {m['key']:<20} {m['accuracy']:>9.4f} {m['f1']:>9.4f} {note:>12}")

    best = max(results, key=lambda x: x["f1"])
    print(f"  → 最优: {best['key']}  (F1={best['f1']:.4f})")


def print_cross_dataset_summary(all_results):
    print(f"\n{'='*80}")
    print(f"{'数据集':<12} {'BiEncoder(Cosine)':>18} {'BiEncoder(Triplet)':>18} {'CrossEncoder':>14}")
    print(f"{'':12} {'F1':>8} {'Acc':>8}  {'F1':>8} {'Acc':>8}  {'F1':>6} {'Acc':>6}")
    print(f"{'-'*80}")

    by_dataset = {}
    for r in all_results:
        by_dataset.setdefault(r["dataset"], {})[r["key"]] = r

    for ds in sorted(by_dataset):
        row = by_dataset[ds]
        def cell(key):
            m = row.get(key)
            return (m["f1"], m["accuracy"]) if m else (None, None)

        c_f1, c_acc = cell("biencoder_cosine")
        t_f1, t_acc = cell("biencoder_triplet")
        x_f1, x_acc = cell("crossencoder")

        def fmt(f1, acc):
            return f"{f1:>8.4f} {acc:>8.4f}" if f1 is not None else f"{'N/A':>8} {'N/A':>8}"

        print(f"  {ds:<10} {fmt(c_f1, c_acc)}  {fmt(t_f1, t_acc)}  {fmt(x_f1, x_acc)}")


def plot_dataset_comparison_bar(all_results, save_path):
    """按数据集分组的方法对比柱状图。"""
    datasets = sorted({r["dataset"] for r in all_results})
    method_keys = [m["key"] for m in METHODS]
    method_labels = [m["label"] for m in METHODS]
    colors = [m["color"] for m in METHODS]

    lookup = {(r["dataset"], r["key"]): r["f1"] for r in all_results}

    x = np.arange(len(datasets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (key, label, color) in enumerate(zip(method_keys, method_labels, colors)):
        f1s = [lookup.get((ds, key), 0) for ds in datasets]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, f1s, width, label=label, color=color, alpha=0.85)
        for bar in bars:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Method Comparison across Datasets (validation)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_comparison_bar(results, dataset_name, save_path):
    """单数据集准确率 / F1 对比柱状图。"""
    names  = [m["label"] for m in results]
    accs   = [m["accuracy"] for m in results]
    f1s    = [m["f1"] for m in results]
    colors = [m["color"] for m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w / 2, f1s, w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Method Comparison on {dataset_name} (validation)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="多数据集文本匹配方法效果对比")
    parser.add_argument("--datasets",   nargs="+", default=["bq_corpus", "lcqmc"],
                        help="待对比的数据集目录名（位于 data/ 下）")
    parser.add_argument("--split",      default="validation",
                        choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  评估集: {args.split}")
    print(f"数据集: {', '.join(args.datasets)}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for dataset_name in args.datasets:
        data_dir = DATA_ROOT / dataset_name
        if not data_dir.exists():
            print(f"\n[SKIP] 数据目录不存在: {data_dir}")
            continue

        print(f"\n{'#'*70}")
        print(f"# 评估数据集: {dataset_name}")
        dataset_results = []

        for m in METHODS:
            print(f"\n  加载 {m['key']} ...")
            metrics = load_and_eval(m, tokenizer, device, data_dir, args.split, args.batch_size)
            if metrics is None:
                continue
            metrics.update({
                "label":   m["label"],
                "color":   m["color"],
                "key":     m["key"],
                "type":    m["type"],
                "dataset": dataset_name,
            })
            dataset_results.append(metrics)
            all_results.append(metrics)

        if dataset_results:
            print_dataset_table(dataset_name, dataset_results)
            plot_comparison_bar(
                dataset_results, dataset_name,
                FIG_DIR / f"method_comparison_{dataset_name}.png",
            )

    if not all_results:
        print("\n没有可用的 checkpoint，请先在各数据集上运行训练脚本，例如：")
        print("  python train_biencoder.py --data_dir data/bq_corpus --loss cosine")
        print("  python train_biencoder.py --data_dir data/lcqmc --loss cosine")
        print("  python train_crossencoder.py --data_dir data/bq_corpus")
        return

    print_cross_dataset_summary(all_results)

    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]
    log_path = LOG_DIR / "dataset_method_comparison.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    plot_dataset_comparison_bar(all_results, FIG_DIR / "dataset_method_comparison.png")


if __name__ == "__main__":
    main()
