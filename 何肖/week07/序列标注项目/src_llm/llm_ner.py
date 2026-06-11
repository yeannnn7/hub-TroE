"""
使用大模型 API 做 NER：zero-shot vs few-shot 对比

使用方式：
  python llm_ner.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPE_ZH = {
    "PER": "人名", "ORG": "组织机构", "LOC": "地名",
    
}

SYSTEM_PROMPT = """你是一个新闻实体识别助手。从文本中识别命名实体，以 JSON 格式输出。
实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地名）
输出格式（严格遵守，不输出其他内容）：
{"entities": [{"text": "实体文本", "type": "实体类型"}]}
无实体时输出：{"entities": []}"""

# 从 train.json 中选取：短句、新闻体、均含 PER/ORG/LOC 三类实体
FEW_SHOT_EXAMPLES = [
    {
        "text": "本报驻埃及记者朱梦魁新华社记者安江",
        "output": '{"entities": [{"text": "埃及", "type": "LOC"}, {"text": "朱梦魁", "type": "PER"}, {"text": "新华社", "type": "ORG"}, {"text": "安江", "type": "PER"}]}',
    },
    {
        "text": "他，就是湖北省阳新县国税局长桂训华。",
        "output": '{"entities": [{"text": "湖北省", "type": "LOC"}, {"text": "阳新县国税局", "type": "ORG"}, {"text": "桂训华", "type": "PER"}]}',
    },
    {
        "text": "伊朗队教练塔勒比：美国人踢得非常好。",
        "output": '{"entities": [{"text": "伊朗队", "type": "ORG"}, {"text": "塔勒比", "type": "PER"}, {"text": "美国", "type": "LOC"}]}',
    },
]

ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())

def build_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise EnvironmentError("请设置环境变量 DEEPSEEK_API_KEY")
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",  
    )


def entity_types_in_record(record: dict) -> set[str]:
    """从 BIO 标签提取该句包含的实体类型。"""
    return {tag[2:] for tag in record["ner_tags"] if tag.startswith("B-")}


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从 validation 分层采样：每类实体先各抽 n/3 条，再随机补齐到 n 条。

    与 evaluate_sft.py 使用相同 seed，保证 LLM 与 SFT 在相同样本上对比。
    """
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)

    by_type = defaultdict(list)
    for r in records:
        for etype in entity_types_in_record(r):
            by_type[etype].append(r)

    selected_ids = set()
    selected_list = []

    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [r for r in by_type[etype] if id(r) not in selected_ids]
        chosen = random.sample(candidates, min(per_type, len(candidates)))
        for r in chosen:
            if len(selected_list) >= n:
                break
            if id(r) not in selected_ids:
                selected_ids.add(id(r))
                selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected_ids]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def gold_spans_from_record(record: dict) -> set[tuple[str, str, int, int]]:
    """从 peoples_daily 的 BIO 格式提取 gold spans。"""
    tokens = record["tokens"]
    ner_tags = record["ner_tags"]
    spans = set()
    i, n = 0, len(tokens)
    while i < n:
        tag = ner_tags[i]
        if tag == "O" or not tag.startswith("B-"):
            i += 1
            continue
        etype = tag[2:]          # "B-LOC" -> "LOC"
        j = i + 1
        while j < n and ner_tags[j] == f"I-{etype}":
            j += 1
        surface = "".join(tokens[i:j])
        start = i
        end = j - 1              # 闭区间，和 CLUENER 的 (5, 7) 一致
        spans.add((surface, etype, start, end))
        i = j
    return spans

def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """调用 LLM API，返回文本输出，带简单重试。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""

def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]

def few_shot_prompt(text: str) -> list[dict]:
    """在 system prompt 后插入 3 组 user/assistant 样例，引导 JSON 格式与边界习惯。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages

def parse_args():
    parser = argparse.ArgumentParser(description="LLM NER 对比评估")
    parser.add_argument("--n_samples", default=100, type=int)
    parser.add_argument("--model", default="deepseek-chat", choices=["deepseek-chat", "deepseek-reasoner"])
    return parser.parse_args()

def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """从 LLM JSON 输出解析 span；type 必须是 PER/ORG/LOC，text 须在原文中可 find。"""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
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
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES_EN:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans

def compute_span_f1(golds: list[set[tuple[str, str, int, int]]], preds: list[set[tuple[str, str, int, int]]]) -> dict:
    """span 级 micro-F1：预测 span 与 gold 在 (text, type, 起止位置) 完全一致才算 TP。"""
    total_gold = sum(len(g) for g in golds)
    total_pred = sum(len(p) for p in preds)
    total_tp = sum(len(g & p) for g, p in zip(golds, preds))
    precision = total_tp / total_pred if total_pred > 0 else 0
    recall = total_tp / total_gold if total_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}
    

def main():
    args = parse_args()
    client = build_client()
    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = "".join(record["tokens"])
        gold = gold_spans_from_record(record)

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"LLM NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }
    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")

if __name__ == "__main__":
    main()