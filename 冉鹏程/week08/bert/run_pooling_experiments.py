"""
批量运行 BiEncoder 池化实验并汇总结果

使用方式：
  python run_pooling_experiments.py
  python run_pooling_experiments.py --loss cosine --pools cls mean max --epochs 1

依赖：
  复用 train_biencoder.py 的训练逻辑，无需重复实现模型与评估。
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
TRAIN_SCRIPT = Path(__file__).with_name("train_biencoder.py")


def parse_args():
    parser = argparse.ArgumentParser(description="批量运行 BiEncoder 池化实验")
    parser.add_argument("--loss", default="cosine", choices=["cosine", "triplet"])
    parser.add_argument("--pools", nargs="+", default=["cls", "mean", "max"],
                        choices=["cls", "mean", "max"])
    parser.add_argument("--bert_path", default=str(BERT_PATH), type=str)
    parser.add_argument("--data_dir", default=str(DATA_DIR), type=str)
    parser.add_argument("--num_hidden_layers", default=4, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_length", default=64, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--head_lr_mult", default=5.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--grad_accum", default=1, type=int)
    parser.add_argument("--margin", default=0.3, type=float)
    parser.add_argument("--device_note", default="", type=str)
    parser.add_argument("--python", default=sys.executable, type=str,
                        help="用于启动 train_biencoder.py 的 Python 解释器")
    return parser.parse_args()


def build_train_command(args, pool):
    return [
        args.python,
        str(TRAIN_SCRIPT),
        "--loss", args.loss,
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
        "--margin", str(args.margin),
    ]


def load_experiment_result(loss, pool):
    ckpt_path = CKPT_DIR / f"biencoder_{loss}_{pool}_best.pt"
    log_path = LOG_DIR / f"biencoder_{loss}_{pool}_log.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"log 不存在: {log_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    with open(log_path, encoding="utf-8") as f:
        log_records = json.load(f)

    best_epoch_record = max(log_records, key=lambda x: x["val_f1"])
    saved_args = ckpt.get("args", {})
    return {
        "pool": pool,
        "loss": loss,
        "best_epoch": ckpt.get("epoch", best_epoch_record["epoch"]),
        "val_acc": ckpt.get("val_acc", best_epoch_record["val_acc"]),
        "val_f1": ckpt.get("val_f1", best_epoch_record["val_f1"]),
        "threshold": ckpt.get("threshold", best_epoch_record.get("threshold")),
        "checkpoint": str(ckpt_path),
        "log_path": str(log_path),
        "train_args": saved_args,
    }


def print_summary(results):
    print(f"\n{'=' * 78}")
    print(f"{'pool':<8} {'loss':<10} {'best_epoch':>10} {'val_acc':>10} {'val_f1':>10} {'threshold':>10}")
    print(f"{'-' * 78}")
    for item in results:
        threshold = item["threshold"]
        threshold_str = "-" if threshold is None else f"{threshold:.2f}"
        print(
            f"{item['pool']:<8} {item['loss']:<10} {item['best_epoch']:>10} "
            f"{item['val_acc']:>10.4f} {item['val_f1']:>10.4f} {threshold_str:>10}"
        )

    best = results[0]
    print(f"\n最佳池化: {best['pool']}  (val_f1={best['val_f1']:.4f}, val_acc={best['val_acc']:.4f})")


def save_summary(args, results):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / f"pooling_comparison_{args.loss}.json"
    payload = {
        "loss": args.loss,
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
            "margin": args.margin,
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
        print(f"\n{'=' * 78}")
        print(f"开始训练池化方式: {pool}")
        print("执行命令:")
        print(" ".join(command))
        subprocess.run(command, check=True)

    results = [load_experiment_result(args.loss, pool) for pool in args.pools]
    results.sort(key=lambda x: x["val_f1"], reverse=True)
    print_summary(results)
    save_summary(args, results)


if __name__ == "__main__":
    main()
