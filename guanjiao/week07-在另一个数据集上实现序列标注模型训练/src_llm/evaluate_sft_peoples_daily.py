import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from peoples_daily_utils import (
    SYSTEM_PROMPT,
    compute_span_f1,
    gold_spans_from_record,
    load_split,
    pred_spans_from_output,
    record_text,
    sample_records,
)


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
MODEL_PATH = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter_peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate peoples_daily SFT NER")
    parser.add_argument("--model_path", default=str(MODEL_PATH))
    parser.add_argument("--ckpt_dir", default=str(ADAPTER_DIR))
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("Loading a LoRA adapter requires peft")
        tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_path).resolve()), trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            str(Path(model_path).resolve()),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer


def generate_ner(text: str, model, tokenizer, device: torch.device, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    model_path = Path(args.model_path)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_dir}")
    if (ckpt_dir / "adapter_config.json").exists() and not model_path.exists():
        raise FileNotFoundError(f"base model not found: {model_path}")

    records = load_split(Path(args.data_dir), args.split)
    n_samples = 5 if args.demo else args.n_samples
    samples = sample_records(records, n_samples, args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} | dataset=peoples_daily | split={args.split} | samples={len(samples)}")
    model, tokenizer = load_model(str(model_path), str(ckpt_dir), device)

    all_golds = []
    all_preds = []
    detail_records = []
    parse_fail = 0
    t0 = time.time()

    for record in tqdm(samples, desc="Evaluate"):
        text = record_text(record)
        gold = gold_spans_from_record(record)
        raw = generate_ner(text, model, tokenizer, device, args.max_new_tokens)
        pred = pred_spans_from_output(text, raw)
        if not re.search(r"\{.*entities.*\}", raw, re.DOTALL):
            parse_fail += 1

        all_golds.append(gold)
        all_preds.append(pred)
        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t, "start": b, "end": e} for s, t, b, e in gold],
            "pred": [{"text": s, "type": t, "start": b, "end": e} for s, t, b, e in pred],
            "raw_output": raw,
        })

    elapsed = time.time() - t0
    metrics = compute_span_f1(all_golds, all_preds)
    result = {
        "model": "Qwen2-0.5B-Instruct SFT",
        "dataset": "peoples_daily",
        "split": args.split,
        "n_samples": len(samples),
        "parse_fail": parse_fail,
        "elapsed_s": elapsed,
        "metrics": metrics,
        "detail": detail_records,
    }

    print("\nLLM SFT peoples_daily result")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"parse_fail: {parse_fail}/{len(samples)}")
    print(f"elapsed: {elapsed:.1f}s")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / f"eval_sft_peoples_daily_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
