"""
人民日报 peoples_daily：字级 BIO → BERT 子词对齐

数据格式（每条）：
  {"tokens": ["海", "钓", ...], "ner_tags": ["O", "B-LOC", "I-LOC", ...]}
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerBase

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

ENTITY_TYPES = ["PER", "ORG", "LOC"]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 7 类 BIO 标签：O + B/I × {PER, ORG, LOC}。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def ner_tags_to_label_ids(
    tokens: list[str], ner_tags: list[str], label2id: dict
) -> list[int]:
    """把与 tokens 等长的 BIO 字符串列表转为 label id。"""
    if len(tokens) != len(ner_tags):
        raise ValueError(
            f"tokens 与 ner_tags 长度不一致: {len(tokens)} vs {len(ner_tags)}"
        )
    return [label2id.get(tag, label2id["O"]) for tag in ner_tags]


class PeoplesDailyDataset(Dataset):
    def __init__(
        self,
        records: list,
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def _get_word_ids(self, encoding, num_chars: int) -> list[int | None]:
        """Fast tokenizer 用 word_ids；慢版 BertTokenizer 按中文一字一 token 对齐。"""
        try:
            return encoding.word_ids(batch_index=0)
        except ValueError:
            ids = encoding["input_ids"].squeeze(0).tolist()
            special = {
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            }
            word_ids: list[int | None] = []
            char_i = 0
            for tid in ids:
                if tid in special:
                    word_ids.append(None)
                elif char_i < num_chars:
                    word_ids.append(char_i)
                    char_i += 1
                else:
                    word_ids.append(None)
            return word_ids

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]
        char_labels = ner_tags_to_label_ids(tokens, ner_tags, self.label2id)

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 字级 BIO → BERT 子词对齐：仅每个字的「首个子词」保留标签，其余标 -100
        # （CrossEntropy / CRF 的 ignore_index=-100，避免对 [CLS]/[SEP]/续字子词算 loss）
        word_ids = self._get_word_ids(encoding, len(char_labels))
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 新字的首个子词：继承该字的 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字的后续子词（WordPiece 被拆成多 token）
                aligned_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: PreTrainedTokenizerBase,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 64,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def collect_bio_sequences(
    labels_batch: list[list[int]],
    pred_ids_batch: list[list[int]],
    id2label: dict[int, str],
    use_crf: bool = False,
) -> tuple[list[list[str]], list[list[str]]]:
    """把 label id / pred id 转为等长 BIO 字符串序列（跳过 -100）。

    Linear / CRF 的 pred 与 input 等长，必须按位置 j 取 pred_ids[i][j]。
    若用顺序游标 pred_idx 递增，会把 [CLS]/子词位置的预测混入实体序列，F1 会接近 0。
    """
    all_golds: list[list[str]] = []
    all_preds: list[list[str]] = []

    for i, token_labels in enumerate(labels_batch):
        gold_seq: list[str] = []
        pred_seq: list[str] = []
        for j, gold_id in enumerate(token_labels):
            if gold_id == -100:
                continue
            gold_seq.append(id2label[gold_id])
            # 与 gold 同一 token 位置对齐取预测（非顺序扫描）
            pid = pred_ids_batch[i][j] if j < len(pred_ids_batch[i]) else 0
            pred_seq.append(id2label.get(pid, "O"))
        all_golds.append(gold_seq)
        all_preds.append(pred_seq)

    return all_golds, all_preds


if __name__ == "__main__":
    _, label2id, id2label = build_label_schema()

    tokens = [
        "海", "钓", "比", "赛", "地", "点", "在", "厦", "门", "与",
        "金", "门", "之", "间", "的", "海", "域", "。",
    ]
    # 与数据一致的 BIO（「海域」应为 B-LOC + I-LOC，不是两个 B-LOC）
    ner_tags = [
        "O", "O", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O",
        "B-LOC", "I-LOC", "O", "O", "O", "B-LOC", "I-LOC", "O",
    ]

    label_ids = ner_tags_to_label_ids(tokens, ner_tags, label2id)
    print("字:", "".join(tokens))
    print("BIO:", ner_tags)
    print("id:", label_ids)
    print("id→标签:", [id2label[i] for i in label_ids])
