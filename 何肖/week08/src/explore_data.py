"""
bq_corpus 数据集探索与可视化

使用方式：
  python explore_data.py
  python explore_data.py --data_dir ../data/bq_corpus --output_dir ../outputs/figures
  python explore_data.py --skip_token          # 跳过 Token 长度分析（较慢）

依赖：
  pip install matplotlib numpy transformers
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "bq_corpus"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
FIG_DIR = ROOT / "outputs" / "figures"


def setup_chinese_font():
    """配置 matplotlib 中文字体（macOS / Windows / Linux 通用回退）。"""
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "STHeiti",
        "Heiti SC",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return name

    for path in (
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).exists():
            font_name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = font_name
            plt.rcParams["axes.unicode_minus"] = False
            return font_name

    plt.rcParams["axes.unicode_minus"] = False
    return None


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_splits(data_dir):
    """从目录加载 train / validation / test 三个 jsonl 文件。"""
    data_dir = Path(data_dir)
    splits = {}
    for split in ["train", "validation", "test"]:
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            splits[split] = load_jsonl(path)
    return splits


# 图 1：标签分布
def plot_label_distribution(splits_data, output_dir):
    fig, axes = plt.subplots(1, len(splits_data), figsize=(10, 4))
    if len(splits_data) == 1:
        axes = [axes]

    for ax, (split_name, rows) in zip(axes, splits_data.items()):
        labels = [r["label"] for r in rows]
        cnt = Counter(labels)
        counts = [cnt.get(0, 0), cnt.get(1, 0)]
        bars = ax.bar(["不相似 (0)", "相似 (1)"], counts,
                      color=["#F44336", "#2196F3"], width=0.5)
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    f"{c}\n({c/len(rows)*100:.1f}%)", ha="center", va="bottom",
                    fontsize=9)
        ax.set_title(f"{split_name}（{len(rows):,} 条）")
        ax.set_ylabel("数量")
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle("bq_corpus 标签分布（正例约 50%，负例约 50%）", fontsize=12, y=1.02)
    fig.tight_layout()
    save_path = output_dir / "label_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")


# 图 2：句子字符长度分布（train，正/负对比）
def plot_char_length(rows, output_dir):
    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def lens(rs):
        return [len(r["sentence1"]) for r in rs] + [len(r["sentence2"]) for r in rs]

    pos_lens = lens(pos_rows)
    neg_lens = lens(neg_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_lens, bins=40, alpha=0.6, label="正样本（相似）",
            color="#2196F3", density=True)
    ax.hist(neg_lens, bins=40, alpha=0.6, label="负样本（不相似）",
            color="#F44336", density=True)
    ax.axvline(32, color="black", linestyle="--", linewidth=1,
               label="max_length=32")
    ax.axvline(64, color="gray", linestyle="--", linewidth=1,
               label="max_length=64")
    ax.set_xlabel("句子字符长度")
    ax.set_ylabel("密度")
    ax.set_title("正/负样本句子长度分布（train）")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / "char_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")

    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]
    print(f"  字符长度统计（train 全部句子）：")
    print(f"    均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}  最长={max(all_lens)}")
    for threshold in [32, 48, 64, 96]:
        cover = sum(1 for l in all_lens if l <= threshold) / len(all_lens) * 100
        print(f"    max_length={threshold:3d} 覆盖率: {cover:.1f}%")


# 图 3：Token 数分布（BERT Tokenizer）
def plot_token_length(rows, tokenizer, output_dir):
    print("  计算 Token 长度（需要 tokenize，稍慢...）")
    token_lens = []
    for r in rows[:5000]:
        t1 = len(tokenizer.tokenize(r["sentence1"]))
        t2 = len(tokenizer.tokenize(r["sentence2"]))
        token_lens.extend([t1, t2])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(token_lens, bins=40, color="#4CAF50", alpha=0.8, density=True)
    ax.axvline(np.mean(token_lens), color="red", linestyle="-",
               label=f"均值={np.mean(token_lens):.1f}")
    ax.axvline(np.percentile(token_lens, 95), color="orange", linestyle="--",
               label=f"P95={np.percentile(token_lens, 95):.0f}")
    ax.set_xlabel("单句 Token 数（不含 [CLS]/[SEP]）")
    ax.set_ylabel("密度")
    ax.set_title("单句 Token 数分布（train 前 5000 条）")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / "token_length_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  Token 长度：均值={np.mean(token_lens):.1f}  "
          f"P95={np.percentile(token_lens, 95):.0f}  最长={max(token_lens)}")


# 图 4：正/负样本长度差（捷径检测）
def plot_length_diff(rows, output_dir):
    pos_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 1]
    neg_diffs = [abs(len(r["sentence1"]) - len(r["sentence2"]))
                 for r in rows if r["label"] == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(pos_diffs, bins=30, alpha=0.6, label=f"正样本 均值={np.mean(pos_diffs):.1f}",
            color="#2196F3", density=True)
    ax.hist(neg_diffs, bins=30, alpha=0.6, label=f"负样本 均值={np.mean(neg_diffs):.1f}",
            color="#F44336", density=True)
    ax.set_xlabel("|len(s1) - len(s2)| 字符数")
    ax.set_ylabel("密度")
    ax.set_title("正/负样本句子长度差分布（length bias 检测）")
    ax.legend()
    fig.tight_layout()

    save_path = output_dir / "length_diff_distribution.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  图表已保存 → {save_path}")
    print(f"  长度差：正样本均值={np.mean(pos_diffs):.1f}  负样本均值={np.mean(neg_diffs):.1f}")
    if np.mean(pos_diffs) < np.mean(neg_diffs) * 0.7:
        print("  ⚠️  正样本长度差明显更小，存在 length bias 风险")
    else:
        print("  ✓ 正/负样本长度差接近，无明显 length bias")


def print_stats(name, rows):
    labels = [r["label"] for r in rows]
    cnt = Counter(labels)
    diffs = [abs(len(r["sentence1"]) - len(r["sentence2"])) for r in rows]
    all_lens = [len(r["sentence1"]) for r in rows] + [len(r["sentence2"]) for r in rows]

    print(f"\n{'='*50}")
    print(f"【{name}】共 {len(rows):,} 条")
    print(f"{'='*50}")
    n_pos = cnt.get(1, 0)
    n_neg = cnt.get(0, 0)
    print(f"  正样本（相似）  : {n_pos:>6,} ({n_pos/len(rows)*100:.1f}%)")
    print(f"  负样本（不相似）: {n_neg:>6,} ({n_neg/len(rows)*100:.1f}%)")
    print(f"  不均衡比 (neg/pos): {n_neg/max(n_pos, 1):.1f}x")
    print(f"  句子字符长度 — 均值={np.mean(all_lens):.1f}  中位数={np.median(all_lens):.0f}  "
          f"P95={np.percentile(all_lens, 95):.0f}")
    print(f"  长度差 — 均值={np.mean(diffs):.1f}  中位数={np.median(diffs):.0f}  "
          f"P95={np.percentile(diffs, 95):.0f}")
    print(f"  示例正样本：")
    for r in [r for r in rows if r["label"] == 1][:2]:
        print(f"    ✓  {r['sentence1']!r}  ||  {r['sentence2']!r}")
    print(f"  示例负样本：")
    for r in [r for r in rows if r["label"] == 0][:2]:
        print(f"    ✗  {r['sentence1']!r}  ||  {r['sentence2']!r}")


def parse_args():
    parser = argparse.ArgumentParser(description="Explore bq_corpus data")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--output_dir", type=str, default=str(FIG_DIR))
    parser.add_argument("--skip_token", action="store_true", help="跳过 Token 长度分析（较慢）")
    return parser.parse_args()


def main():
    args = parse_args()
    font_name = setup_chinese_font()
    if font_name:
        print(f"使用中文字体: {font_name}")
    else:
        print("警告: 未找到中文字体，图中文字可能显示为方框")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits_data = load_splits(args.data_dir)
    if not splits_data:
        print(f"未找到数据文件，请检查目录: {args.data_dir}")
        print("期望存在: train.jsonl, validation.jsonl, test.jsonl")
        return

    for name, rows in splits_data.items():
        print_stats(name, rows)

    train_rows = splits_data.get("train", [])
    if not train_rows:
        print("train.jsonl 不存在")
        return

    print(f"\n{'='*50}")
    print("生成可视化图表...")
    plot_label_distribution(splits_data, output_dir)
    plot_char_length(train_rows, output_dir)
    plot_length_diff(train_rows, output_dir)

    if not args.skip_token:
        tokenizer = BertTokenizer.from_pretrained(
            str(Path(args.bert_path).resolve()), local_files_only=False
        )
        plot_token_length(train_rows, tokenizer, output_dir)

    print(f"\n所有图表已保存至 → {output_dir}")


if __name__ == "__main__":
    main()
