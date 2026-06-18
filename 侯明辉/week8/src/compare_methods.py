import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
# Windows 默认 GBK 编码无法打印 unicode 符号,强制 UTF-8
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except (AttributeError, Exception):
        pass

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
from evaluate import eval_biencoder, eval_crossencoder, plot_similarity_distribution
from model import build_biencoder, build_crossencoder

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data" / "lcqmc"
BERT_PATH  = ROOT / "pretrain_models" / "bert-base-chinese"
CKPT_DIR   = ROOT / "outputs" / "checkpoints"
FIG_DIR    = ROOT / "outputs" / "figures"
LOG_DIR    = ROOT / "outputs" / "logs"


def build_methods(dataset_tag):
    """根据数据集标签构造方法列表（ckpt 命名规则与训练脚本一致）"""
    return [
        {
            "key":   f"biencoder_cosine_{dataset_tag}",
            "label": f"BiEncoder\n(Cosine)\n[{dataset_tag}]",
            "ckpt":  f"biencoder_cosine_{dataset_tag}_best.pt",
            "type":  "biencoder",
            "color": "#2196F3",
        },
        {
            "key":   f"biencoder_triplet_{dataset_tag}",
            "label": f"BiEncoder\n(Triplet)\n[{dataset_tag}]",
            "ckpt":  f"biencoder_triplet_{dataset_tag}_best.pt",
            "type":  "biencoder",
            "color": "#4CAF50",
        },
        {
            "key":   f"crossencoder_{dataset_tag}",
            "label": f"CrossEncoder\n(CrossEntropy)\n[{dataset_tag}]",
            "ckpt":  f"crossencoder_{dataset_tag}_best.pt",
            "type":  "crossencoder",
            "color": "#FF9800",
        },
    ]


def load_and_eval(method, tokenizer, device, data_dir, split, batch_size):
    ckpt_path = CKPT_DIR / method["ckpt"]
    if not ckpt_path.exists():
        print(f"  [SKIP] checkpoint 不存在: {ckpt_path}")
        return None

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get("args", {})

    if method["type"] == "biencoder":
        model = build_biencoder(
            bert_path=str(BERT_PATH),
            pool=saved.get("pool", "mean"),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds     = PairDataset(data_path, tokenizer, max_length=saved.get("max_length", 64))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_biencoder(model, loader, device)

    else:  # crossencoder
        model = build_crossencoder(
            bert_path=str(BERT_PATH),
            num_hidden_layers=saved.get("num_hidden_layers"),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        data_path = Path(data_dir) / f"{split}.jsonl"
        ds     = CrossEncoderDataset(data_path, tokenizer, max_length=128)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        metrics = eval_crossencoder(model, loader, device)

    metrics["model"] = model
    metrics["ckpt"]  = ckpt
    return metrics


# ── 可视化 ────────────────────────────────────────────────────────────────

def plot_comparison_bar(results, dataset_tag, save_path):
    """准确率 / F1 对比柱状图"""
    names    = [m["label"].replace("\n", " ") for m in results]
    accs     = [m["accuracy"] for m in results]
    f1s      = [m["f1"]       for m in results]
    colors   = [m["color"]    for m in results]

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color=colors, alpha=0.85)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1 (weighted)", color=colors, alpha=0.5,
                   hatch="//", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(f"Method Comparison on {dataset_tag} Validation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


def plot_sim_distributions(biencoder_results, dataset_tag, save_path):
    """所有 BiEncoder 方法的相似度分布叠放对比"""
    n = len(biencoder_results)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, m in zip(axes, biencoder_results):
        sims   = np.array(m["similarities"])
        labels = np.array(m["labels"])
        ax.hist(sims[labels==1], bins=40, alpha=0.6, label="positive", color="#2196F3", density=True)
        ax.hist(sims[labels==0], bins=40, alpha=0.6, label="negative", color="#F44336", density=True)
        ax.axvline(m["threshold"], color="black", linestyle="--",
                   label=f"threshold={m['threshold']:.2f}")
        ax.set_title(m["label"].replace("\n", " "))
        ax.set_xlabel("Cosine Similarity")
        ax.legend(fontsize=8)

    fig.suptitle(f"BiEncoder Similarity Distribution on {dataset_tag}", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# ── 主流程 ────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="三种文本匹配方法效果对比")
    parser.add_argument("--data_dir",   default=str(DATA_DIR), type=str,
                        help="数据集目录")
    parser.add_argument("--split",      default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", default=64, type=int)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_tag = Path(args.data_dir).name
    methods     = build_methods(dataset_tag)

    print(f"设备: {device}  数据集: {dataset_tag}  评估集: {args.split}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 逐方法评估 ────────────────────────────────────────────────────────
    all_results = []
    for m in methods:
        print(f"\n{'='*55}")
        print(f"加载 {m['key']} ...")
        metrics = load_and_eval(m, tokenizer, device, args.data_dir, args.split, args.batch_size)
        if metrics is None:
            continue
        metrics.update({"label": m["label"], "color": m["color"], "key": m["key"],
                        "type": m["type"]})
        all_results.append(metrics)

    if not all_results:
        print(f"\n没有可用的 checkpoint，请先运行训练脚本。")
        print(f"提示：训练时使用 --data_dir {args.data_dir} 可生成匹配的 checkpoint 文件名。")
        return

    # ── 控制台对比表 ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'方法':<35} {'Accuracy':>9} {'F1(weighted)':>13} {'额外信息':>15}")
    print(f"{'-'*70}")
    for m in all_results:
        extra = (f"threshold={m['threshold']:.2f}"
                 if m["type"] == "biencoder" else "argmax")
        print(f"  {m['key']:<33} {m['accuracy']:>9.4f} {m['f1']:>13.4f} {extra:>15}")

    print(f"\n{'─'*70}")
    print("结论速览：")
    best_acc = max(all_results, key=lambda x: x["accuracy"])
    best_f1  = max(all_results, key=lambda x: x["f1"])
    print(f"  最高 Accuracy : {best_acc['key']} ({best_acc['accuracy']:.4f})")
    print(f"  最高 F1       : {best_f1['key']}  ({best_f1['f1']:.4f})")

    bi_results = [m for m in all_results if m["type"] == "biencoder"]
    if len(bi_results) == 2:
        a, b = bi_results
        delta_acc = b["accuracy"] - a["accuracy"]
        delta_f1  = b["f1"] - b["f1"] if False else b["f1"] - a["f1"]
        print(f"\n  Cosine vs Triplet (Δ):")
        print(f"    Accuracy: {delta_acc:+.4f}  F1: {delta_f1:+.4f}")
        if abs(delta_f1) < 0.01:
            print("    → 两种 Loss 差距不大")
        elif delta_f1 > 0:
            print("    → TripletLoss 更优（大数据集上相对距离约束更有效）")
        else:
            print("    → CosineEmbeddingLoss 更优（直接对标签优化更稳定）")

    # ── 保存对比日志 ──────────────────────────────────────────────────────
    SKIP_KEYS = {"model", "similarities", "labels", "logits", "ckpt"}

    def _to_py(v):
        if hasattr(v, "item"):
            return v.item()
        return v

    log = [{k: _to_py(v) for k, v in m.items() if k not in SKIP_KEYS}
           for m in all_results]
    log_path = LOG_DIR / f"method_comparison_{dataset_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"\n对比日志 → {log_path}")

    # ── 可视化 ────────────────────────────────────────────────────────────
    plot_comparison_bar(all_results, dataset_tag,
                        FIG_DIR / f"method_comparison_bar_{dataset_tag}.png")

    bi_with_sim = [m for m in all_results if m["type"] == "biencoder" and "similarities" in m]
    if bi_with_sim:
        plot_sim_distributions(bi_with_sim, dataset_tag,
                               FIG_DIR / f"biencoder_sim_distributions_{dataset_tag}.png")


if __name__ == "__main__":
    main()