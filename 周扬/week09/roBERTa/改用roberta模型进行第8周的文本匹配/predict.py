"""
文本匹配交互预测脚本

用法：
  python predict.py
  python predict.py --text_a "今天天气不错" --text_b "今天的天气很好"
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

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


def parse_args():
    parser = argparse.ArgumentParser(description='文本匹配预测')
    parser.add_argument('--model_dir', type=str, default=str(CURRENT_DIR / 'checkpoints'),
                        help='模型目录')
    parser.add_argument('--max_length', type=int, default=32,
                        help='最大序列长度，默认优先使用训练时配置')
    default_device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--device', type=str, default=default_device,
                        help='设备')
    parser.add_argument('--text_a', type=str, default='',
                        help='句子 A；如果不传，就进入交互模式')
    parser.add_argument('--text_b', type=str, default='',
                        help='句子 B；如果不传，就进入交互模式')
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


def encode_text(tokenizer, text: str, max_length: int, device: str):
    encoded = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    return {k: v.to(device) for k, v in encoded.items()}


def predict_pair(model, tokenizer, model_type: str, text_a: str, text_b: str, max_length: int, device: str):
    enc_a = encode_text(tokenizer, text_a, max_length, device)
    enc_b = encode_text(tokenizer, text_b, max_length, device)

    with torch.no_grad():
        if model_type == 'biencoder':
            outputs = model(
                enc_a['input_ids'], enc_a['attention_mask'], enc_a['token_type_ids'],
                enc_b['input_ids'], enc_b['attention_mask'], enc_b['token_type_ids'],
            )
            similarity = float(outputs['similarity'].item())
            pred_label = 1 if similarity > 0 else 0
            confidence = (similarity + 1.0) / 2.0
            return {
                'pred_label': pred_label,
                'pred_text': '相似' if pred_label == 1 else '不相似',
                'score_name': 'cosine_similarity',
                'score': similarity,
                'confidence_name': 'mapped_confidence',
                'confidence': confidence,
            }

        outputs = model(
            enc_a['input_ids'], enc_a['attention_mask'], enc_a['token_type_ids'],
            enc_b['input_ids'], enc_b['attention_mask'], enc_b['token_type_ids'],
        )
        logits = outputs['logits']
        probs = F.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
        return {
            'pred_label': pred_label,
            'pred_text': '相似' if pred_label == 1 else '不相似',
            'score_name': 'similar_prob',
            'score': float(probs[1].item()),
            'confidence_name': 'max_prob',
            'confidence': float(probs[pred_label].item()),
            'prob_not_similar': float(probs[0].item()),
            'prob_similar': float(probs[1].item()),
        }


def print_result(text_a: str, text_b: str, result: dict, model_type: str):
    print('\n' + '=' * 60)
    print(f'句子A: {text_a}')
    print(f'句子B: {text_b}')
    print(f'模型类型: {model_type}')
    print(f'预测结果: {result["pred_text"]} (label={result["pred_label"]})')
    print(f'{result["score_name"]}: {result["score"]:.4f}')
    print(f'{result["confidence_name"]}: {result["confidence"]:.4f}')

    if 'prob_not_similar' in result:
        print(f'不相似概率: {result["prob_not_similar"]:.4f}')
        print(f'相似概率: {result["prob_similar"]:.4f}')

    if model_type == 'biencoder':
        print('判断规则: similarity > 0 记为相似')
    print('=' * 60)


def interactive_loop(model, tokenizer, model_type: str, max_length: int, device: str):
    print('\n进入交互模式，输入 quit / exit 结束。')

    while True:
        text_a = input('\n请输入句子A: ').strip()
        if text_a.lower() in {'quit', 'exit'}:
            break
        if not text_a:
            print('句子A 不能为空。')
            continue

        text_b = input('请输入句子B: ').strip()
        if text_b.lower() in {'quit', 'exit'}:
            break
        if not text_b:
            print('句子B 不能为空。')
            continue

        result = predict_pair(model, tokenizer, model_type, text_a, text_b, max_length, device)
        print_result(text_a, text_b, result, model_type)


def main():
    args = parse_args()
    model_dir = _resolve_path(args.model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f'找不到模型目录: {model_dir}')

    saved_cfg = load_saved_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model, model_type = load_model(model_dir, saved_cfg, args.device)
    max_length = saved_cfg.get('max_length', args.max_length)

    print('预测配置:')
    print(f'  模型目录: {model_dir}')
    print(f'  模型类型: {model_type}')
    print(f'  Transformer层数: {saved_cfg.get("num_hidden_layers", 3)}')
    print(f'  max_length: {max_length}')
    print(f'  设备: {args.device}')

    if args.text_a and args.text_b:
        result = predict_pair(model, tokenizer, model_type, args.text_a, args.text_b, max_length, args.device)
        print_result(args.text_a, args.text_b, result, model_type)
        return

    interactive_loop(model, tokenizer, model_type, max_length, args.device)


if __name__ == '__main__':
    main()
