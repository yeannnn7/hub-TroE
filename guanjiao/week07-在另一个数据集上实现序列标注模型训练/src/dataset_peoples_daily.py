"""
人民日报 NER 数据集类：token + BIO 标签 → BERT 子词对齐

与 cluener 数据集的关键区别：
  1. 人民日报数据已经是 token + BIO 标签格式，无需 span→BIO 转换
     - tokens:   ["海", "钓", ..., "厦", "门", ...]
     - ner_tags: ["O",  "O",  ..., "B-LOC", "I-LOC", ...]
  2. 实体类型只有 3 类（PER/ORG/LOC），共 7 个 BIO 标签
  3. 复用 BERT 子词对齐逻辑（word_ids 策略）

使用方式：
  from dataset_peoples_daily import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

ENTITY_TYPES = ["PER", "ORG", "LOC"]


def build_label_schema(
    data_dir: Optional[Path] = None,
) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建标签体系，返回 (labels, label2id, id2label)。"""
    d = data_dir or DATA_DIR
    label_file = d / "label_names.json"

    if label_file.exists():
        with open(label_file, "r", encoding="utf-8") as f:
            labels = json.load(f)
    else:
        labels = ["O"]
        for etype in ENTITY_TYPES:
            labels.append(f"B-{etype}")
            labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据流程：
      tokens (字符列表) + ner_tags (BIO 字符串列表)
           → BertTokenizer (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizerFast,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        char_label_ids = [self.label2id.get(tag, 0) for tag in ner_tags]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_label_ids):
                    aligned_labels.append(char_label_ids[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizerFast,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
