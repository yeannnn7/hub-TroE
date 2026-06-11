import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peoples_daily_utils import SYSTEM_PROMPT, load_split, record_text, record_to_target


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"


class PeoplesDailySFTDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int = 256):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": record_text(record)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = (
            self.tokenizer.encode(record_to_target(record), add_special_tokens=False)
            + [self.tokenizer.eos_token_id]
        )

        input_ids = (prompt_ids + response_ids)[: self.max_length]
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_id: int) -> dict:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list = []
    labels_list = []
    mask_list = []
    for item in batch:
        n_tokens = item["input_ids"].size(0)
        pad_len = max_len - n_tokens
        input_ids_list.append(
            torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])
        )
        labels_list.append(
            torch.cat([
                item["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
        )
        mask_list.append(
            torch.cat([
                torch.ones(n_tokens, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
        )
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="SFT Qwen for peoples_daily NER")
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--num_train", type=int, default=-1)
    parser.add_argument("--num_val", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--full_ft", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / (
        "sft_full_ckpt_peoples_daily" if args.full_ft else "sft_adapter_peoples_daily"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_records = load_split(data_dir, "train")
    val_records = load_split(data_dir, "validation")
    if args.num_train > 0:
        train_records = random.sample(train_records, min(args.num_train, len(train_records)))
    if args.num_val > 0:
        val_records = val_records[: args.num_val]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "full fine-tuning" if args.full_ft else "LoRA"
    print(f"device={device} | mode={mode}")
    print(f"dataset=peoples_daily | train={len(train_records)} | val={len(val_records)}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"model_path not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path.resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = PeoplesDailySFTDataset(train_records, tokenizer, args.max_length)
    val_dataset = PeoplesDailySFTDataset(val_records, tokenizer, args.max_length)
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path.resolve()),
        dtype=torch.float32,
        trust_remote_code=True,
    )

    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA training requires peft")
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
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // max(args.grad_accum, 1)
    print(
        f"steps={total_steps} | batch={args.batch_size} | "
        f"grad_accum={args.grad_accum} | epochs={args.epochs} | lr={args.lr}"
    )

    best_val_loss = float("inf")
    log_records = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if len(train_loader) % args.grad_accum != 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / max(total_tokens, 1)
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens

        avg_val_loss = val_loss / max(val_tokens, 1)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | {elapsed:.0f}s"
        )
        log_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  saved best checkpoint -> {ckpt_dir} (val_loss={avg_val_loss:.4f})")

    log_path = output_dir / "logs" / "train_sft_peoples_daily.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"training done | best_val_loss={best_val_loss:.4f}")
    print(f"log -> {log_path}")
    print(f"checkpoint -> {ckpt_dir}")


if __name__ == "__main__":
    main()
