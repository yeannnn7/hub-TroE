"""
在 LCQMC / BQ Corpus 上快速对比不同文本匹配方法。

这个脚本不依赖 BERT checkpoint，适合作为跨数据集实验的快速基线：
  1. Majority：始终预测训练集多数类
  2. LengthSimilarity：句子长度越接近越可能相似
  3. CharJaccard：字符集合重叠
  4. TFIDF-Cosine：字符 n-gram TF-IDF 余弦相似度

阈值只在 train 集上搜索，再到 validation/test 上评估。
"""

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = OUTPUT_DIR / "reports"
FIG_DIR = OUTPUT_DIR / "figures"


def load_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def labels(rows):
    return np.array([int(r["label"]) for r in rows], dtype=int)


def dataset_stats(rows):
    y = labels(rows)
    cnt = Counter(y.tolist())
    return {
        "n": len(rows),
        "positive": cnt.get(1, 0),
        "negative": cnt.get(0, 0),
        "positive_ratio": cnt.get(1, 0) / max(len(rows), 1),
    }


def length_similarity(rows):
    return np.array([
        1.0 / (1.0 + abs(len(r["sentence1"]) - len(r["sentence2"])))
        for r in rows
    ])


def char_jaccard(rows):
    scores = []
    for r in rows:
        a = set(r["sentence1"].replace(" ", ""))
        b = set(r["sentence2"].replace(" ", ""))
        union = len(a | b)
        scores.append(len(a & b) / union if union else 0.0)
    return np.array(scores)


def fit_tfidf(train_rows, max_features, ngram_min, ngram_max):
    corpus = []
    for r in train_rows:
        corpus.append(r["sentence1"])
        corpus.append(r["sentence2"])
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        min_df=2,
        max_features=max_features,
        norm="l2",
    )
    vectorizer.fit(corpus)
    return vectorizer


def tfidf_cosine(rows, vectorizer):
    s1 = [r["sentence1"] for r in rows]
    s2 = [r["sentence2"] for r in rows]
    x1 = vectorizer.transform(s1)
    x2 = vectorizer.transform(s2)
    return np.asarray(x1.multiply(x2).sum(axis=1)).ravel()


def find_best_threshold(scores, y_true):
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo
    best_f1, best_t = -1.0, lo
    for t in np.linspace(lo, hi, 201):
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, average="weighted", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def evaluate_predictions(y_true, pred):
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "f1_positive": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "precision_positive": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "recall_positive": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
    }


def evaluate_score_method(method, train_scores, train_y, eval_scores, eval_y):
    threshold = find_best_threshold(train_scores, train_y)
    pred = (eval_scores >= threshold).astype(int)
    metrics = evaluate_predictions(eval_y, pred)
    metrics.update({"method": method, "threshold": threshold})
    return metrics


def run_dataset(dataset, eval_split, args):
    data_dir = DATA_ROOT / dataset
    train_rows = load_jsonl(data_dir / "train.jsonl", limit=args.train_limit)
    eval_rows = load_jsonl(data_dir / f"{eval_split}.jsonl", limit=args.eval_limit)
    train_y = labels(train_rows)
    eval_y = labels(eval_rows)

    print(f"\n{dataset}: train={len(train_rows):,}, {eval_split}={len(eval_rows):,}")
    results = []

    majority = Counter(train_y.tolist()).most_common(1)[0][0]
    majority_pred = np.full_like(eval_y, majority)
    majority_metrics = evaluate_predictions(eval_y, majority_pred)
    majority_metrics.update({"method": "majority", "threshold": None})
    results.append(majority_metrics)

    train_len = length_similarity(train_rows)
    eval_len = length_similarity(eval_rows)
    results.append(evaluate_score_method("length_similarity", train_len, train_y, eval_len, eval_y))

    train_jaccard = char_jaccard(train_rows)
    eval_jaccard = char_jaccard(eval_rows)
    results.append(evaluate_score_method("char_jaccard", train_jaccard, train_y, eval_jaccard, eval_y))

    print(f"  fitting TF-IDF char {args.ngram_min}-{args.ngram_max}, max_features={args.max_features:,} ...")
    vectorizer = fit_tfidf(train_rows, args.max_features, args.ngram_min, args.ngram_max)
    train_tfidf = tfidf_cosine(train_rows, vectorizer)
    eval_tfidf = tfidf_cosine(eval_rows, vectorizer)
    tfidf_metrics = evaluate_score_method("tfidf_cosine", train_tfidf, train_y, eval_tfidf, eval_y)
    tfidf_metrics["vocab_size"] = len(vectorizer.vocabulary_)
    results.append(tfidf_metrics)

    for item in results:
        item.update({
            "dataset": dataset,
            "eval_split": eval_split,
            "train_stats": dataset_stats(train_rows),
            "eval_stats": dataset_stats(eval_rows),
        })
        print(
            f"  {item['method']:<18} "
            f"acc={item['accuracy']:.4f} "
            f"f1w={item['f1_weighted']:.4f} "
            f"f1_pos={item['f1_positive']:.4f}"
        )
    return results


def save_json(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def save_csv(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset", "eval_split", "method", "accuracy", "f1_weighted",
        "f1_positive", "precision_positive", "recall_positive",
        "threshold", "vocab_size",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


def save_report(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 另外两个文本匹配数据集方法对比",
        "",
        "阈值在 train 集上搜索，表中指标在目标评估集上计算。",
        "",
        "| 数据集 | 方法 | Accuracy | F1(weighted) | F1(pos) | 阈值 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        threshold = "" if r["threshold"] is None else f"{r['threshold']:.4f}"
        lines.append(
            f"| {r['dataset']}:{r['eval_split']} | {r['method']} | "
            f"{r['accuracy']:.4f} | {r['f1_weighted']:.4f} | "
            f"{r['f1_positive']:.4f} | {threshold} |"
        )

    lines.extend(["", "## 结论速览", ""])
    for dataset in sorted({r["dataset"] for r in results}):
        subset = [r for r in results if r["dataset"] == dataset]
        best = max(subset, key=lambda x: x["f1_weighted"])
        jaccard = next(r for r in subset if r["method"] == "char_jaccard")
        tfidf = next(r for r in subset if r["method"] == "tfidf_cosine")
        lines.append(
            f"- {dataset}: 最优方法是 {best['method']}，"
            f"F1(weighted)={best['f1_weighted']:.4f}。"
            f"TF-IDF 相比字符 Jaccard 的 F1 增量为 "
            f"{tfidf['f1_weighted'] - jaccard['f1_weighted']:+.4f}。"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"{r['dataset']}\n{r['method']}" for r in results]
    scores = [r["f1_weighted"] for r in results]
    colors = ["#7C3AED" if r["method"] == "tfidf_cosine" else "#2563EB" for r in results]

    fig, ax = plt.subplots(figsize=(max(9, len(results) * 0.8), 5))
    bars = ax.bar(np.arange(len(results)), scores, color=colors, alpha=0.85)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("Lightweight Method Comparison on LCQMC / BQ Corpus")
    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="LCQMC / BQ Corpus 轻量方法对比")
    parser.add_argument("--datasets", nargs="+", default=["lcqmc", "bq_corpus"])
    parser.add_argument("--eval_split", default="validation", choices=["validation", "test"])
    parser.add_argument("--train_limit", type=int, default=None,
                        help="只取前 N 条训练样本；默认使用完整 train")
    parser.add_argument("--eval_limit", type=int, default=None,
                        help="只取前 N 条评估样本；默认使用完整 split")
    parser.add_argument("--max_features", type=int, default=50000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    all_results = []
    for dataset in args.datasets:
        all_results.extend(run_dataset(dataset, args.eval_split, args))

    suffix = f"{'_'.join(args.datasets)}_{args.eval_split}"
    json_path = LOG_DIR / f"other_datasets_baselines_{suffix}.json"
    csv_path = LOG_DIR / f"other_datasets_baselines_{suffix}.csv"
    report_path = REPORT_DIR / f"other_datasets_baselines_{suffix}.md"
    fig_path = FIG_DIR / f"other_datasets_baselines_{suffix}.png"

    save_json(all_results, json_path)
    save_csv(all_results, csv_path)
    save_report(all_results, report_path)
    plot_results(all_results, fig_path)

    print(f"\nJSON 结果 → {json_path}")
    print(f"CSV 结果  → {csv_path}")
    print(f"报告      → {report_path}")
    print(f"图表      → {fig_path}")


if __name__ == "__main__":
    main()
