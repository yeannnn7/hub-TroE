"""
文本匹配任务训练脚本
"""

import argparse
import time
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from dataset import PairDataset
from model import BiEncoder, BiEncoderForClassification


CURRENT_DIR = Path(__file__).parent
DEFAULT_MODEL_NAME = str((CURRENT_DIR.parents[3] / 'roBERTa' / 'chinese-roberta-wwm-ext-large').resolve())


class TrainLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.touch(exist_ok=True)

    def log(self, message: str = "", use_tqdm: bool = False):
        text = str(message)
        if use_tqdm:
            tqdm.write(text)
        else:
            print(text)
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


def _format_seconds(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    h = seconds_int // 3600
    m = (seconds_int % 3600) // 60
    s = seconds_int % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _get_lr(optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (CURRENT_DIR / p).resolve()


def _maybe_limit_dataset(dataset, limit: int, seed: int, name: str, logger: TrainLogger | None = None):
    log_fn = logger.log if logger is not None else print
    if limit is None or limit <= 0:
        return dataset
    if limit >= len(dataset):
        log_fn(f"{name} 不抽样（limit={limit} >= 数据集大小={len(dataset)}）")
        return dataset
    rnd = random.Random(seed)
    indices = list(range(len(dataset)))
    rnd.shuffle(indices)
    keep = indices[:limit]
    keep.sort()
    from torch.utils.data import Subset
    log_fn(f"{name} 抽样: {len(dataset)} -> {len(keep)} (seed={seed})")
    return Subset(dataset, keep)


def parse_args():
    parser = argparse.ArgumentParser(description='文本匹配训练')

    # 数据路径
    parser.add_argument('--data_dir', type=str, default=str(CURRENT_DIR / 'data' / 'lcqmc'),
                        help='数据集目录')
    parser.add_argument('--train_file', type=str, default='train.jsonl',
                        help='训练集文件名')
    parser.add_argument('--val_file', type=str, default='validation.jsonl',
                        help='验证集文件名')
    parser.add_argument('--train_limit', type=int, default=0,
                        help='训练集抽样数量（0 表示不抽样）')
    parser.add_argument('--val_limit', type=int, default=0,
                        help='验证集抽样数量（0 表示不抽样）')

    # 模型参数
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='预训练模型名称')
    parser.add_argument('--model_type', type=str, default='biencoder',
                        choices=['biencoder', 'classifier'],
                        help='模型类型')
    parser.add_argument('--hidden_size', type=int, default=0,
                        help='隐藏层维度，0 表示从预训练模型配置自动读取')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                        help='实际使用的 BERT Transformer 层数，默认 3 层')

    # 训练参数
    parser.add_argument('--max_length', type=int, default=32,
                        help='最大序列长度')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='预热比例')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='梯度裁剪')

    # 损失函数参数
    parser.add_argument('--loss_type', type=str, default='cosine',
                        choices=['cosine', 'triplet', 'ce'],
                        help='损失函数类型')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='TripletLoss 的 margin')

    # 其他
    parser.add_argument('--output_dir', type=str, default=str(CURRENT_DIR / 'checkpoints'),
                        help='模型保存目录')
    parser.add_argument('--log_dir', type=str, default=str(CURRENT_DIR / 'logs'),
                        help='训练日志目录')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='保存步数')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='日志步数')
    default_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=default_device,
                        help='设备')

    return parser.parse_args()


def load_data(args, tokenizer, logger: TrainLogger):
    """加载数据集"""
    data_dir = _resolve_path(args.data_dir)
    train_path = data_dir / args.train_file
    val_path = data_dir / args.val_file

    if not train_path.exists():
        raise FileNotFoundError(
            f"找不到训练集文件: {train_path}\n"
            f"当前 data_dir={data_dir}\n"
            f"你可以用参数指定正确目录，例如：\n"
            f"  python train.py --data_dir {CURRENT_DIR / 'data' / 'lcqmc'}"
        )
    if not val_path.exists():
        raise FileNotFoundError(
            f"找不到验证集文件: {val_path}\n"
            f"当前 data_dir={data_dir}\n"
            f"你可以用参数指定正确目录，例如：\n"
            f"  python train.py --data_dir {CURRENT_DIR / 'data' / 'lcqmc'}"
        )

    logger.log("加载数据集...")
    train_dataset = PairDataset(
        train_path,
        tokenizer,
        max_length=args.max_length
    )
    val_dataset = PairDataset(
        val_path,
        tokenizer,
        max_length=args.max_length
    )

    train_dataset = _maybe_limit_dataset(train_dataset, args.train_limit, seed=42, name="训练集", logger=logger)
    val_dataset = _maybe_limit_dataset(val_dataset, args.val_limit, seed=43, name="验证集", logger=logger)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    logger.log(f"  训练集: {len(train_dataset)} 条, {len(train_loader)} batch")
    logger.log(f"  验证集: {len(val_dataset)} 条, {len(val_loader)} batch")

    return train_loader, val_loader


def build_model(args, logger: TrainLogger):
    """构建模型"""
    logger.log(f"加载模型: {args.model_name}")
    resolved_hidden_size = args.hidden_size
    if resolved_hidden_size <= 0:
        resolved_hidden_size = int(AutoConfig.from_pretrained(args.model_name).hidden_size)
        logger.log(f"  自动读取隐藏层维度: {resolved_hidden_size}")
    args.hidden_size = resolved_hidden_size

    if args.model_type == 'biencoder':
        model = BiEncoder(args.model_name, args.hidden_size, args.num_hidden_layers)
    else:
        model = BiEncoderForClassification(args.model_name, args.hidden_size, args.num_hidden_layers)

    model = model.to(args.device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log(f"  总参数量: {total_params:,}")
    logger.log(f"  可训练参数: {trainable_params:,}")

    return model


def build_loss_fn(args, logger: TrainLogger):
    """构建损失函数"""
    if args.loss_type == 'cosine':
        # CosineEmbeddingLoss: label=1 表示相似，label=-1 表示不相似
        # 数据集 label 是 0/1，需要转换
        loss_fn = nn.CosineEmbeddingLoss(margin=args.margin)
        logger.log(f"损失函数: CosineEmbeddingLoss(margin={args.margin})")

    elif args.loss_type == 'triplet':
        loss_fn = nn.TripletMarginLoss(margin=args.margin)
        logger.log(f"损失函数: TripletMarginLoss(margin={args.margin})")

    elif args.loss_type == 'ce':
        loss_fn = nn.CrossEntropyLoss()
        logger.log("损失函数: CrossEntropyLoss")

    return loss_fn


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, args, epoch, logger: TrainLogger):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_steps = 0
    total_correct = 0
    total_count = 0
    start_time = time.perf_counter()

    progress_bar = tqdm(
        train_loader,
        desc=f'Epoch {epoch + 1}/{args.num_epochs} [train]',
        dynamic_ncols=True
    )

    for step, batch in enumerate(progress_bar):
        # 移动到设备
        batch = {k: v.to(args.device) for k, v in batch.items()}

        # 前向传播
        if args.model_type == 'biencoder':
            outputs = model(
                batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b']
            )

            if args.loss_type == 'cosine':
                # CosineEmbeddingLoss 需要 label 为 -1 或 1
                # 数据集 label 是 0/1，转换：0 -> -1, 1 -> 1
                target = batch['label'].float()
                target[target == 0] = -1
                loss = loss_fn(outputs['emb_a'], outputs['emb_b'], target)

            elif args.loss_type == 'triplet':
                # TripletLoss 需要 (anchor, positive, negative)
                # 这里简化处理，用同一 batch 内的负样本
                loss = loss_fn(outputs['emb_a'], outputs['emb_b'], outputs['emb_b'][torch.randperm(len(outputs['emb_b']))])

        else:  # classifier
            outputs = model(
                batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b'],
                labels=batch['label']
            )
            loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

        optimizer.step()
        scheduler.step()

        # 统计
        total_loss += loss.item()
        total_steps += 1
        batch_size = int(batch['label'].shape[0])
        total_count += batch_size

        acc = None
        if args.model_type == 'biencoder':
            sim = outputs.get('similarity')
            if sim is None:
                sim = F.cosine_similarity(outputs['emb_a'], outputs['emb_b'])
            pred = (sim > 0).long()
            total_correct += (pred == batch['label']).sum().item()
            acc = total_correct / max(1, total_count)
        else:
            pred = outputs['logits'].argmax(dim=-1)
            total_correct += (pred == batch['label']).sum().item()
            acc = total_correct / max(1, total_count)

        # 更新进度条
        elapsed = time.perf_counter() - start_time
        it_s = total_steps / max(1e-9, elapsed)
        sp_s = total_count / max(1e-9, elapsed)
        lr = _get_lr(optimizer)
        progress_bar.set_postfix({
            'lr': f'{lr:.2e}',
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / total_steps:.4f}',
            'acc': f'{acc:.4f}',
            'it/s': f'{it_s:.2f}',
            'sp/s': f'{sp_s:.0f}',
        })

        # 日志
        if (step + 1) % args.logging_steps == 0:
            eta = (len(train_loader) - (step + 1)) / max(1e-9, it_s)
            logger.log(
                f"train | epoch {epoch + 1}/{args.num_epochs} | step {step + 1}/{len(train_loader)} | "
                f"lr={lr:.2e} loss={loss.item():.4f} avg_loss={total_loss / total_steps:.4f} "
                f"acc={acc:.4f} it/s={it_s:.2f} sp/s={sp_s:.0f} "
                f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}",
                use_tqdm=True
            )

    avg_loss = total_loss / total_steps
    avg_acc = total_correct / max(1, total_count)
    epoch_time = time.perf_counter() - start_time
    logger.log(
        f"train | epoch {epoch + 1}/{args.num_epochs} done | "
        f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f} time={_format_seconds(epoch_time)}",
        use_tqdm=True
    )

    return avg_loss, avg_acc


def evaluate(model, val_loader, loss_fn, args, logger: TrainLogger):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='eval', dynamic_ncols=True)
        for step, batch in enumerate(progress_bar):
            # 移动到设备
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # 前向传播
            if args.model_type == 'biencoder':
                outputs = model(
                    batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                    batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b']
                )

                if args.loss_type == 'cosine':
                    target = batch['label'].float()
                    target[target == 0] = -1
                    loss = loss_fn(outputs['emb_a'], outputs['emb_b'], target)

                elif args.loss_type == 'triplet':
                    loss = loss_fn(
                        outputs['emb_a'],
                        outputs['emb_b'],
                        outputs['emb_b'][torch.randperm(len(outputs['emb_b']))],
                    )

                sim = outputs.get('similarity')
                if sim is None:
                    sim = F.cosine_similarity(outputs['emb_a'], outputs['emb_b'])
                pred = (sim > 0).long()
                correct += (pred == batch['label']).sum().item()

            else:  # classifier
                outputs = model(
                    batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                    batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b'],
                    labels=batch['label']
                )
                loss = outputs['loss']

                # 计算准确率
                pred = outputs['logits'].argmax(dim=-1)
                correct += (pred == batch['label']).sum().item()

            total_loss += loss.item()
            total += len(batch['label'])
            elapsed = time.perf_counter() - start_time
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / max(1, step + 1):.4f}',
                'acc': f'{(correct / max(1, total)):.4f}',
                'sp/s': f'{(total / max(1e-9, elapsed)):.0f}',
            })

            if (step + 1) % args.logging_steps == 0:
                logger.log(
                    f"eval | step {step + 1}/{len(val_loader)} | "
                    f"loss={loss.item():.4f} avg_loss={total_loss / max(1, step + 1):.4f} "
                    f"acc={(correct / max(1, total)):.4f} sp/s={(total / max(1e-9, elapsed)):.0f}",
                    use_tqdm=True
                )

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    elapsed = time.perf_counter() - start_time
    logger.log(f'eval done | loss={avg_loss:.4f} acc={accuracy:.4f} time={_format_seconds(elapsed)}')

    return avg_loss, accuracy


def save_model(model, tokenizer, output_dir, args, logger: TrainLogger):
    """保存模型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型
    if args.model_type == 'biencoder':
        model.bert.save_pretrained(output_dir)
    else:
        model.encoder.bert.save_pretrained(output_dir)
        torch.save(model.classifier.state_dict(), output_dir / 'classifier.pt')

    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)

    # 保存配置
    config = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'max_length': args.max_length,
        'loss_type': args.loss_type,
        'margin': args.margin,
    }
    import json
    with open(output_dir / 'train_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    logger.log(f'模型已保存到: {output_dir}')


def build_logger(args) -> TrainLogger:
    log_dir = _resolve_path(args.log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'train_{timestamp}.log'
    return TrainLogger(log_path)


def main():
    args = parse_args()
    logger = build_logger(args)

    # 设置随机种子
    torch.manual_seed(42)
    logger.log(f'日志文件: {logger.log_path}')

    # 加载 tokenizer
    logger.log(f"加载 tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # 加载数据
    train_loader, val_loader = load_data(args, tokenizer, logger)

    # 构建模型
    model = build_model(args, logger)

    # 构建损失函数
    loss_fn = build_loss_fn(args, logger)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    logger.log('\n训练配置:')
    logger.log(f'  设备: {args.device}')
    logger.log(f'  批次大小: {args.batch_size}')
    logger.log(f'  学习率: {args.learning_rate}')
    logger.log(f'  Transformer层数: {args.num_hidden_layers}')
    logger.log(f'  训练轮数: {args.num_epochs}')
    logger.log(f'  总步数: {total_steps}')
    logger.log(f'  预热步数: {warmup_steps}')
    logger.log(f'  日志目录: {_resolve_path(args.log_dir)}')
    logger.log()

    # 训练循环
    best_val_acc = 0

    for epoch in range(args.num_epochs):
        logger.log(f'\n{"=" * 50}')
        logger.log(f'Epoch {epoch + 1}/{args.num_epochs}')
        logger.log(f'{"=" * 50}\n')

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, args, epoch, logger)

        # 评估
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, args, logger)
        logger.log(
            f'epoch {epoch + 1}/{args.num_epochs} summary | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, tokenizer, args.output_dir, args, logger)
            logger.log(f'保存最佳模型，验证集准确率: {val_acc:.4f}')

    logger.log(f'\n训练完成！最佳验证集准确率: {best_val_acc:.4f}')


if __name__ == '__main__':
    main()
