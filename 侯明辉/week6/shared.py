"""
homework 共用工具模块

提供三种训练方法共享的常量、数据加载和评估函数。

使用方式：
  from shared import DATA_DIR, BERT_PATH, QWEN_PATH, LABEL_NAMES, NUM_LABELS
  from shared import load_raw_data, evaluate_discriminative, evaluate_bert
  from shared import classify_one_qwen, parse_qwen_prediction, set_seed
"""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════════════════

ROOT         = Path(__file__).parent.parent   # text_classification项目/
DATA_DIR     = ROOT / "data"
BERT_PATH    = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
QWEN_PATH    = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
OUTPUT_DIR   = ROOT / "outputs" / "compare_demo"

LABEL_NAMES = [
    "故事", "文化", "娱乐", "体育", "财经",
    "房产", "汽车", "教育", "科技", "军事",
    "旅游", "国际", "证券", "农业", "电竞",
]
NUM_LABELS = 15

# Windows 多进程 OpenMP 冲突规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def set_seed(seed=42):
    """设置全局随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载（通用）
# ══════════════════════════════════════════════════════════════════════════════

def load_raw_data(num_train=-1, num_val=1000, seed=42):
    """加载 JSON 数据，返回 train / val 列表。"""
    with open(DATA_DIR / "train.json", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(DATA_DIR / "val.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if num_train > 0:
        random.seed(seed)
        train_raw = random.sample(train_raw, min(num_train, len(train_raw)))
    if num_val > 0:
        val_raw = val_raw[:num_val]

    print(f"数据加载完成: train={len(train_raw)}, val={len(val_raw)}")
    return train_raw, val_raw


# ══════════════════════════════════════════════════════════════════════════════
# 评估函数
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_discriminative(model, loader, device):
    """评估判别式模型（LSTM 等），返回 (accuracy, macro_f1)。"""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["label"]
            logits    = model(input_ids)
            preds     = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


def evaluate_bert(model, loader, device):
    """评估 BERT 模型，返回 (accuracy, macro_f1)。"""
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["label"]
            logits = model(input_ids, attention_mask, token_type_ids)
            preds  = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc      = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, macro_f1


# ══════════════════════════════════════════════════════════════════════════════
# Qwen2 生成式分类工具
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(LABEL_NAMES)
)


def classify_one_qwen(text, model, tokenizer, device, max_new_tokens=8):
    """用 Qwen2 模型生成一条分类结果（返回原始输出字符串）。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"新闻标题：{text}\n类别："},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_qwen_prediction(raw_output):
    """模糊匹配：模型可能输出"科技"或"科技新闻"，只检查是否包含类别名。"""
    for name in LABEL_NAMES:
        if name in raw_output:
            return name
    return None