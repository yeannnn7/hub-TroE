"""
文本匹配任务评估脚本

支持两种模型：
  1. BiEncoder
  2. BiEncoderForClassification

使用方式：
  python evaluate.py --model_dir checkpoints --split val
  python evaluate.py --model_dir checkpoints --split test
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from tqdm import tqdm

from dataset import PairDataset
from model import BiEncoder, BiEncoderForClassification


CURRENT_DIR = Path(__file__).parent
DEFAULT_MODEL_NAME = str((CURRENT_DIR.parents[3] / 'roBERTa' / 'chinese-roberta-wwm-ext-large').resolve())


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (CURRENT_DIR / p).resolve()


def _load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _format_seconds(seconds: float) -> str:
    seconds_int = max(0, int(seconds))
    h = seconds_int // 3600
    m = (seconds_int % 3600) // 60
    s = seconds_int % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def parse_args():
    parser = argparse.ArgumentParser(description='文本匹配评估')
    parser.add_argument('--data_dir', type=str, default=str(CURRENT_DIR / 'data' / 'lcqmc'),
                        help='数据集目录')
    parser.add_argument('--val_file', type=str, default='validation.jsonl',
                        help='验证集文件名')
    parser.add_argument('--test_file', type=str, default='test.jsonl',
                        help='测试集文件名')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'],
                        help='评估验证集还是测试集')
    parser.add_argument('--model_dir', type=str, default=str(CURRENT_DIR / 'checkpoints'),
                        help='模型目录')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='评估批次大小，默认优先使用训练时配置')
    parser.add_argument('--max_length', type=int, default=32,
                        help='最大序列长度，默认优先使用训练时配置')
    default_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=default_device,
                        help='设备')
    return parser.parse_args()


def load_saved_config(model_dir: Path) -> dict:
    train_config_path = model_dir / 'train_config.json'
    if train_config_path.exists():
        return _load_json(train_config_path)

    legacy_config_path = model_dir / 'config.json'
    if legacy_config_path.exists():
        legacy_config = _load_json(legacy_config_path)
        if 'model_name' in legacy_config and 'model_type' in legacy_config:
            return legacy_config

    raise FileNotFoundError(
        f'找不到训练配置文件: {train_config_path}；'
        f'旧版兼容配置也不存在或格式不正确: {legacy_config_path}'
    )


def build_eval_loader(args, tokenizer, saved_cfg):
    data_dir = _resolve_path(args.data_dir)
    file_name = args.val_file if args.split == 'val' else args.test_file
    data_path = data_dir / file_name

    if not data_path.exists():
        raise FileNotFoundError(f'找不到评估文件: {data_path}')

    max_length = saved_cfg.get('max_length', args.max_length)
    batch_size = args.batch_size

    dataset = PairDataset(data_path, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f'加载评估集: {data_path}')
    print(f'  样本数: {len(dataset)}')
    print(f'  batch 数: {len(loader)}')
    print(f'  max_length: {max_length}')
    print(f'  batch_size: {batch_size}')

    return loader


def load_model(model_dir: Path, saved_cfg: dict, device: str):
    base_model_name = saved_cfg.get('model_name', DEFAULT_MODEL_NAME)
    model_type = saved_cfg.get('model_type', 'biencoder')
    hidden_size = saved_cfg.get('hidden_size', 0)
    num_hidden_layers = saved_cfg.get('num_hidden_layers', 3)
    if hidden_size <= 0:
        hidden_size = int(AutoConfig.from_pretrained(base_model_name).hidden_size)

    if model_type == 'biencoder':
        model = BiEncoder(
            str(model_dir),
            hidden_size,
            num_hidden_layers=num_hidden_layers,
            config_name=base_model_name,
        )
    else:
        model = BiEncoderForClassification(
            str(model_dir),
            hidden_size,
            num_hidden_layers=num_hidden_layers,
            config_name=base_model_name,
        )
        classifier_path = model_dir / 'classifier.pt'
        if not classifier_path.exists():
            raise FileNotFoundError(f'找不到分类头参数: {classifier_path}')
        model.classifier.load_state_dict(torch.load(classifier_path, map_location=device))

    model = model.to(device)
    model.eval()
    return model, model_type


def build_loss_fn(saved_cfg):
    loss_type = saved_cfg.get('loss_type', 'cosine')
    margin = saved_cfg.get('margin', 0.5)

    if loss_type == 'cosine':
        return nn.CosineEmbeddingLoss(margin=margin), loss_type
    if loss_type == 'triplet':
        return nn.TripletMarginLoss(margin=margin), loss_type
    return nn.CrossEntropyLoss(), 'ce'


def evaluate(model, dataloader, loss_fn, model_type, loss_type, device):
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    start_time = time.perf_counter()

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='eval', dynamic_ncols=True)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            if model_type == 'biencoder':
                outputs = model(
                    batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                    batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b']
                )

                if loss_type == 'cosine':
                    target = batch['label'].float()
                    target[target == 0] = -1
                    loss = loss_fn(outputs['emb_a'], outputs['emb_b'], target)
                elif loss_type == 'triplet':
                    loss = loss_fn(
                        outputs['emb_a'],
                        outputs['emb_b'],
                        outputs['emb_b'][torch.randperm(len(outputs['emb_b']))],
                    )
                else:
                    target = batch['label'].float()
                    target[target == 0] = -1
                    loss = nn.CosineEmbeddingLoss()(outputs['emb_a'], outputs['emb_b'], target)

                sim = outputs.get('similarity')
                if sim is None:
                    sim = F.cosine_similarity(outputs['emb_a'], outputs['emb_b'])
                pred = (sim > 0).long()

            else:
                outputs = model(
                    batch['input_ids_a'], batch['attention_mask_a'], batch['token_type_ids_a'],
                    batch['input_ids_b'], batch['attention_mask_b'], batch['token_type_ids_b'],
                    labels=batch['label']
                )
                loss = outputs['loss']
                pred = outputs['logits'].argmax(dim=-1)

            batch_size = int(batch['label'].shape[0])
            total_loss += loss.item()
            total_correct += (pred == batch['label']).sum().item()
            total_count += batch_size

            elapsed = time.perf_counter() - start_time
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / max(1, progress_bar.n + 1):.4f}',
                'acc': f'{total_correct / max(1, total_count):.4f}',
                'sp/s': f'{total_count / max(1e-9, elapsed):.0f}',
            })

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_count)
    elapsed = time.perf_counter() - start_time
    return avg_loss, accuracy, elapsed


def main():
    args = parse_args()
    model_dir = _resolve_path(args.model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f'找不到模型目录: {model_dir}')

    saved_cfg = load_saved_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    dataloader = build_eval_loader(args, tokenizer, saved_cfg)
    model, model_type = load_model(model_dir, saved_cfg, args.device)
    loss_fn, loss_type = build_loss_fn(saved_cfg)

    print('\n评估配置:')
    print(f'  模型目录: {model_dir}')
    print(f'  模型类型: {model_type}')
    print(f'  Transformer层数: {saved_cfg.get("num_hidden_layers", 3)}')
    print(f'  损失函数: {loss_type}')
    print(f'  评估数据: {args.split}')
    print(f'  设备: {args.device}')
    print()

    avg_loss, accuracy, elapsed = evaluate(
        model=model,
        dataloader=dataloader,
        loss_fn=loss_fn,
        model_type=model_type,
        loss_type=loss_type,
        device=args.device,
    )

    print(f'eval done | split={args.split} loss={avg_loss:.4f} acc={accuracy:.4f} time={_format_seconds(elapsed)}')


if __name__ == '__main__':
    main()
