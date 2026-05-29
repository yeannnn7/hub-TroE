"""
Qwen2-0.5B-Instruct SFT (LoRA) 文本分类训练

教学重点：
  1. 指令微调格式：把分类任务转化为 system/user/assistant chat 格式
  2. Loss masking：只在 assistant 输出部分（类别名）计算 cross-entropy loss
     prompt 部分 labels 全设为 -100，PyTorch 自动忽略
  3. LoRA 原理：冻结原始权重，旁路低秩矩阵 ΔW = B·A，仅训练约 0.5% 参数
  4. 生成式分类 vs 判别式分类：需要 generate 解码，推理效率低于 BERT

使用方式：
  python train_llm.py                            # LoRA 微调，5000 条快速演示
  python train_llm.py --num_train -1             # LoRA 微调，全部 53K 条
  python train_llm.py --epochs 1                 # 快速验证流程
  python train_llm.py --lora_r 16                # 增大 LoRA rank，更多参数

依赖：
  pip install torch transformers peft scikit-learn tqdm
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from shared import (
    DATA_DIR, QWEN_PATH, OUTPUT_DIR, NUM_LABELS, LABEL_NAMES,
    SYSTEM_PROMPT, load_raw_data, classify_one_qwen, parse_qwen_prediction, set_seed,
)


# ══════════════════════════════════════════════════════════════════════════════
# SFT Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """把分类数据转换为 chat-format 的 SFT 训练样本（loss masking）。"""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = LABEL_NAMES[item["label"]]

        # 构建 prompt（system + user），用 tokenize=False 拿到文本串再 encode
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": "新闻标题：" + item["sentence"] + "\n类别："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # response = 类别名 + EOS
        response_ids = (
            self.tokenizer.encode(label_name, add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        # 拼接完整序列，截断至 max_length
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # loss mask：prompt 部分全设 -100
        labels = ([-100] * prompt_len + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def sft_collate_fn(batch, pad_id):
    """右填充使同批次序列等长。"""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n   = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([
            item["input_ids"],
            torch.full((pad,), pad_id, dtype=torch.long)
        ]))
        labels_list.append(torch.cat([
            item["labels"],
            torch.full((pad,), -100, dtype=torch.long)
        ]))
        mask_list.append(torch.cat([
            torch.ones(n, dtype=torch.long),
            torch.zeros(pad, dtype=torch.long)
        ]))

    return {
        "input_ids":      torch.stack(input_ids_list),
        "labels":         torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 训练入口
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2-0.5B-Instruct SFT (LoRA) 文本分类训练"
    )
    parser.add_argument("--num_train",    default=5000,  type=int,
                        help="训练样本数，-1 使用全部")
    parser.add_argument("--num_val",      default=1000,  type=int,
                        help="验证样本数")
    parser.add_argument("--batch_size",   default=4,     type=int,
                        help="SFT batch size（LLM 显存需求更大）")
    parser.add_argument("--max_length",   default=128,   type=int,
                        help="序列最大长度")
    parser.add_argument("--epochs",       default=2,     type=int)
    parser.add_argument("--lr",           default=2e-4,  type=float,
                        help="学习率（LoRA 推荐 2e-4）")
    parser.add_argument("--grad_accum",   default=4,     type=int,
                        help="梯度累积步数，等效 batch = batch_size × grad_accum")
    parser.add_argument("--lora_r",       default=8,     type=int,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha",   default=16,    type=int,
                        help="LoRA 缩放因子，有效学习率 ≈ lr × alpha/r")
    parser.add_argument("--num_eval_generate", default=100, type=int,
                        help="每个 epoch 用 generate 评估的样本数（较慢）")
    parser.add_argument("--qwen_path",    default=str(QWEN_PATH))
    parser.add_argument("--seed",         default=42,    type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    if not PEFT_AVAILABLE:
        print("错误：LoRA 模式需要 peft 库，请运行: pip install peft>=0.14.0")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ── 加载数据 ────────────────────────────────────────────────────────────
    train_raw, val_raw = load_raw_data(args.num_train, args.num_val, args.seed)

    # ── Tokenizer ───────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.qwen_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.qwen_path).resolve()),
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 数据集 ──────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:200], tokenizer, args.max_length)

    _collate = lambda b: sft_collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, collate_fn=_collate)

    # ── 模型 ────────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.qwen_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.qwen_path).resolve()),
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    n_params    = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    # ── 优化器 ──────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps} (batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs})")

    # ── 训练循环 ────────────────────────────────────────────────────────────
    log_records = []
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]",
                     leave=False)
        for step, batch in enumerate(pbar):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens    = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # --- 验证 loss ---
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens   = (labels != -100).sum().item()
                val_loss  += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        # --- 验证分类准确率（generate 评估） ---
        sample_val = val_raw[:args.num_eval_generate]
        correct, total = 0, 0
        for item in tqdm(sample_val, desc="Eval (generate)", leave=False):
            raw = classify_one_qwen(item["sentence"], model, tokenizer, device)
            pred_name = parse_qwen_prediction(raw)
            true_name = LABEL_NAMES[item["label"]]
            if pred_name is not None:
                if pred_name == true_name:
                    correct += 1
                total += 1
        val_acc = correct / max(total, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | "
              f"val_acc(generate)={val_acc:.4f} ({correct}/{total}) | "
              f"{elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "val_acc": val_acc,
            "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # ── 保存结果 ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "method": "Qwen2-0.5B SFT (LoRA)",
        "params_M": n_params,
        "trainable_M": n_trainable,
        "trainable_pct": n_trainable / n_params * 100,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "epochs": args.epochs,
        "log": log_records,
    }
    with open(OUTPUT_DIR / "llm_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成。最优 val_acc={best_val_acc:.4f}, 最优 val_loss={best_val_loss:.4f}")
    print(f"结果已保存 → {OUTPUT_DIR / 'llm_result.json'}")


if __name__ == "__main__":
    main()
