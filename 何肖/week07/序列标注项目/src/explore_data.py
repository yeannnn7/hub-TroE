"""
peoples_daily 数据集探索与可视化

数据格式（download_data.py 生成）：
  {"tokens": ["在", "这", ...], "ner_tags": ["O", "B-PER", "I-PER", ...]}

输出：
  outputs/figures/entity_distribution.png
  outputs/figures/text_length_distribution.png
  outputs/figures/entity_length_distribution.png
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
FIG_DIR = ROOT / "outputs" / "figures"

ET_LABEL = {"PER": "人名", "ORG": "机构", "LOC": "地名"}


def load_split(split: str) -> list:
    path = DATA_DIR / f"{split}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bio_to_entities(tokens: list[str], ner_tags: list[str]) -> list[dict]:
    """从字级 BIO 序列解析实体：B-X 开实体，连续 I-X 拼接至下一个非 I-X。"""
    entities = []
    i = 0
    n = len(tokens)
    while i < n:
        tag = ner_tags[i]
        if tag == "O" or not tag.startswith("B-"):
            i += 1
            continue
        etype = tag[2:]  # PER / ORG / LOC
        j = i + 1
        while j < n and ner_tags[j] == f"I-{etype}":
            j += 1
        text = "".join(tokens[i:j])
        entities.append({"text": text, "type": etype})
        i = j
    return entities


def collect_stats(records: list) -> dict:
    entity_type_counts = Counter()
    entity_lengths = []
    text_lengths = []
    entities_by_type = {}

    for record in records:
        tokens = record["tokens"]
        ner_tags = record["ner_tags"]
        text_lengths.append(len(tokens))

        for ent in bio_to_entities(tokens, ner_tags):
            etype = ent["type"]
            entity_type_counts[etype] += 1
            entity_lengths.append(len(ent["text"]))
            entities_by_type.setdefault(etype, []).append(ent["text"])

    return {
        "entity_type_counts": entity_type_counts,
        "entity_lengths": entity_lengths,
        "text_lengths": text_lengths,
        "entities_by_type": entities_by_type,
    }


def print_summary(stats_train: dict, stats_val: dict):
    print("=" * 60)
    print("peoples_daily 数据集统计摘要")
    print("=" * 60)
    for name, stats in [("训练集", stats_train), ("验证集", stats_val)]:
        tl = stats["text_lengths"]
        el = stats["entity_lengths"]
        print(f"\n【{name}】")
        print(f"  句子数：{len(tl)}")
        print(f"  文本平均长度：{sum(tl) / len(tl):.1f} 字")
        print(f"  实体总数：{sum(stats['entity_type_counts'].values())}")
        if el:
            print(f"  实体平均长度：{sum(el) / len(el):.1f} 字")
        print("  各类实体频次：")
        for etype, cnt in sorted(stats["entity_type_counts"].items(), key=lambda x: -x[1]):
            cn = ET_LABEL.get(etype, etype)
            print(f"    {etype} ({cn}): {cnt}")


def plot_entity_distribution(stats: dict, split_name: str):
    counts = stats["entity_type_counts"]
    labels = [f"{k}\n({ET_LABEL.get(k, k)})" for k in sorted(counts)]
    values = [counts[k] for k in sorted(counts)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="#4C72B0", alpha=0.85)
    ax.set_title(f"peoples_daily 各类实体频次（{split_name}）")
    ax.set_ylabel("实体数量")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"entity_distribution_{split_name}.png", dpi=120)
    plt.close()


def plot_text_length_distribution(stats: dict, split_name: str):
    lengths = stats["text_lengths"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=50, color="#4C72B0", alpha=0.8)
    ax.set_title(f"文本长度分布（{split_name}）")
    ax.set_xlabel("字符数（tokens 数）")
    ax.set_ylabel("句子数")
    fig.savefig(FIG_DIR / f"text_length_{split_name}.png", dpi=120)
    plt.close()


def plot_entity_length_distribution(stats: dict, split_name: str):
    lengths = stats["entity_lengths"]
    if not lengths:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=range(1, min(20, max(lengths) + 2)), color="#55A868", alpha=0.85)
    ax.set_title(f"实体长度分布（{split_name}）")
    ax.set_xlabel("实体字符数")
    ax.set_ylabel("出现次数")
    fig.savefig(FIG_DIR / f"entity_length_{split_name}.png", dpi=120)
    plt.close()


def main():
    parse_args()

    train_records = load_split("train")
    val_records = load_split("validation")

    stats_train = collect_stats(train_records)
    stats_val = collect_stats(val_records)

    print_summary(stats_train, stats_val)

    print("\n正在生成可视化图表...")
    plot_entity_distribution(stats_train, "train")
    plot_text_length_distribution(stats_train, "train")
    plot_entity_length_distribution(stats_train, "train")

    print(f"\n探索完成！图表已保存到 {FIG_DIR}/")


def parse_args():
    parser = argparse.ArgumentParser(description="探索人民日报 NER 数据集")
    return parser.parse_args()


if __name__ == "__main__":
    main()
