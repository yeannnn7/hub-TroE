import json
import random
import re
from pathlib import Path


ENTITY_TYPES = ["PER", "ORG", "LOC"]

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构名",
    "LOC": "地名",
}

SYSTEM_PROMPT = (
    "你是一个中文命名实体识别助手。请从用户输入文本中识别实体，"
    "只输出 JSON，不要输出解释文字。\n"
    "实体类型：PER（人名）、ORG（组织机构名）、LOC（地名）。\n"
    '输出格式：{"entities": [{"text": "实体文本", "type": "PER/ORG/LOC"}]}\n'
    '如果没有实体，输出：{"entities": []}'
)


def record_text(record: dict) -> str:
    return "".join(record["tokens"])


def bio_to_entities(tokens: list[str], tags: list[str]) -> list[dict]:
    entities = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if not tag.startswith("B-"):
            i += 1
            continue

        etype = tag[2:]
        start = i
        i += 1
        while i < len(tokens) and tags[i] == f"I-{etype}":
            i += 1
        end = i - 1

        if etype in ENTITY_TYPES:
            entities.append({
                "text": "".join(tokens[start : end + 1]),
                "type": etype,
                "start": start,
                "end": end,
            })
    return entities


def record_to_target(record: dict) -> str:
    entities = [
        {"text": ent["text"], "type": ent["type"]}
        for ent in bio_to_entities(record["tokens"], record["ner_tags"])
    ]
    return json.dumps({"entities": entities}, ensure_ascii=False)


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    return {
        (ent["text"], ent["type"], ent["start"], ent["end"])
        for ent in bio_to_entities(record["tokens"], record["ner_tags"])
    }


def pred_spans_from_output(text: str, raw_output: str) -> set[tuple[str, str, int, int]]:
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return set()

    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()

    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()

    spans = set()
    used_ranges: set[tuple[int, int]] = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES:
            continue

        start = find_unused_span(text, surface, used_ranges)
        if start < 0:
            continue
        end = start + len(surface) - 1
        used_ranges.add((start, end))
        spans.add((surface, etype, start, end))
    return spans


def find_unused_span(text: str, surface: str, used_ranges: set[tuple[int, int]]) -> int:
    start = 0
    while True:
        idx = text.find(surface, start)
        if idx < 0:
            return -1
        end = idx + len(surface) - 1
        if (idx, end) not in used_ranges:
            return idx
        start = idx + 1


def compute_span_f1(all_golds: list[set], all_preds: list[set]) -> dict:
    tp = sum(len(gold & pred) for gold, pred in zip(all_golds, all_preds))
    pred_total = sum(len(pred) for pred in all_preds)
    gold_total = sum(len(gold) for gold in all_golds)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "pred_total": pred_total,
        "gold_total": gold_total,
    }


def load_split(data_dir: Path, split: str) -> list[dict]:
    with open(data_dir / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def sample_records(records: list[dict], n_samples: int, seed: int) -> list[dict]:
    if n_samples <= 0 or n_samples >= len(records):
        return records
    random.seed(seed)
    return random.sample(records, n_samples)
