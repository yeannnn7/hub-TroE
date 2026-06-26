"""
按顺序运行 CrossEncoder 的三种池化实验并汇总结果 🙆🏻‍♀️

使用方式：
  python run_crossencoder_pooling_experiments.py
  python run_crossencoder_pooling_experiments.py --epochs 3 --eval_steps 500
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
MODEL_ROOT = Path("/root/rpc_stu/model")
DATA_DIR = ROOT / "data" / "bq_corpus"
BERT_PATH = MODEL_ROOT / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
TRAIN_SCRIPT = Path(__file__).with_name("train_crossencoder_pooling.py")


def parse_args():
    parser = argparse.ArgumentParser(description="顺序运行 CrossEncoder 三种池化实验")
    parser.add_argument("--pools", nargs="+", default=["cls", "mean", "max"],
                        choices=["cls", "mean", "max"])
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--num_hidden_layers", default=12, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=128, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--eval_steps", default=500, type=int)
    parser.add_argument("--python", default=sys.executable, type=str)
    parser.add_argument("--device_note", default="", type=str)
    return parser.parse_args()


def build_train_command(args, pool):
    return [
        args.python,
        str(TRAIN_SCRIPT),
        "--pool", pool,
        "--bert_path", args.bert_path,
        "--data_dir", args.data_dir,
        "--num_hidden_layers", str(args.num_hidden_layers),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--max_length", str(args.max_length),
        "--lr", str(args.lr),
        "--head_lr_mult", str(args.head_lr_mult),
        "--warmup_ratio", str(args.warmup_ratio),
        "--grad_accum", str(args.grad_accum),
        "--eval_steps", str(args.eval_steps),
    ]


def load_experiment_result(pool):
    ckpt_path = CKPT_DIR / f"crossencoder_{pool}_best.pt"
    log_path = LOG_DIR / f"crossencoder_{pool}_log.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"log 不存在: {log_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(log_path, encoding="utf-8") as f:
        log_records = json.load(f)

    best_record = max(log_records, key=lambda x: x["val_f1"])
    return {
        "pool": pool,
        "best_epoch": ckpt.get("epoch", 0),
        "best_step": ckpt.get("step", best_record["step"]),
        "val_loss": ckpt.get("val_loss", best_record["val_loss"]),
        "val_acc": ckpt.get("val_acc", best_record["val_acc"]),
        "val_f1": ckpt.get("val_f1", best_record["val_f1"]),
        "checkpoint": str(ckpt_path),
        "log_path": str(log_path),
        "train_args": ckpt.get("args", {}),
    }


def print_summary(results):
    print(f"\n{'=' * 86}")
    print(f"{'pool':<8} {'best_epoch':>10} {'best_step':>10} {'val_loss':>10} {'val_acc':>10} {'val_f1':>10}")
    print(f"{'-' * 86}")
    for item in results:
        print(
            f"{item['pool']:<8} {item['best_epoch']:>10} {item['best_step']:>10} "
            f"{item['val_loss']:>10.4f} {item['val_acc']:>10.4f} {item['val_f1']:>10.4f}"
        )

    best = results[0]
    print(f"\n最佳池化🙆🏻‍♀️: {best['pool']}  (val_f1={best['val_f1']:.4f}, val_loss={best['val_loss']:.4f})")


def save_summary(args, results):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "crossencoder_pooling_comparison.json"
    payload = {
        "pools": args.pools,
        "device_note": args.device_note,
        "train_config": {
            "bert_path": args.bert_path,
            "data_dir": args.data_dir,
            "num_hidden_layers": args.num_hidden_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "lr": args.lr,
            "head_lr_mult": args.head_lr_mult,
            "warmup_ratio": args.warmup_ratio,
            "grad_accum": args.grad_accum,
            "eval_steps": args.eval_steps,
        },
        "results": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"汇总结果已保存 → {summary_path}")


def main():
    args = parse_args()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for pool in args.pools:
        command = build_train_command(args, pool)
        print(f"\n{'=' * 86}")
        print(f"开始训练 CrossEncoder 池化方式: {pool} 🙆🏻‍♀️")
        print("执行命令:")
        print(" ".join(command))
        subprocess.run(command, check=True)

    results = [load_experiment_result(pool) for pool in args.pools]
    results.sort(key=lambda x: x["val_f1"], reverse=True)
    print_summary(results)
    save_summary(args, results)


if __name__ == "__main__":
    main()
