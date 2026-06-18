"""
BQ Corpus文本匹配训练脚本

基于Qwen2.5-0.5B-Instruct的文本匹配训练
支持LoRA微调
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import BQCorpusDatasetForCausal, collate_fn_causal
from model import load_model_and_tokenizer


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=1):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc='Training')
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / accumulation_steps
        
        # 反向传播
        loss.backward()
        
        # 梯度累积
        if (step + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': loss.item() * accumulation_steps})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, tokenizer, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    decoded_results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            # 生成响应
            outputs = model.generate_response(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10
            )
            
            # 解码
            for i in range(len(outputs)):
                generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
                
                # 提取答案
                if "assistant\n" in generated_text:
                    answer = generated_text.split("assistant\n")[-1].strip()
                else:
                    answer = generated_text.strip()
                
                # 判断答案
                pred = 1 if "是" in answer else 0
                true_label = labels[i]
                
                predictions.append(pred)
                true_labels.append(true_label)
                
                decoded_results.append({
                    'sentence1': batch['sentence1'][i],
                    'sentence2': batch['sentence2'][i],
                    'generated_text': answer,
                    'prediction': pred,
                    'label': true_label,
                    'correct': pred == true_label
                })
    
    # 计算指标
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics, decoded_results


def main(args):
    """主训练函数"""
    print("=" * 80)
    print("BQ Corpus文本匹配训练 - 基于Qwen2.5-0.5B-Instruct")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / '..' / 'data' / 'bq_corpus'
    output_dir = base_dir / 'outputs' / 'llm'
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    model.to(device)
    
    # 创建数据集
    print("\n加载数据集...")
    train_dataset = BQCorpusDatasetForCausal(
        str(data_dir / 'train.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    val_dataset = BQCorpusDatasetForCausal(
        str(data_dir / 'validation.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    test_dataset = BQCorpusDatasetForCausal(
        str(data_dir / 'test.jsonl'),
        tokenizer,
        max_length=args.max_len
    )
    
    # 随机采样数据集
    if args.sample_ratio < 1.0:
        print(f"\n随机采样数据集 (采样比例: {args.sample_ratio})...")
        random.seed(args.seed)
        
        original_train_size = len(train_dataset)
        original_val_size = len(val_dataset)
        original_test_size = len(test_dataset)
        
        train_size = int(len(train_dataset) * args.sample_ratio)
        train_indices = random.sample(range(len(train_dataset)), train_size)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        val_size = int(len(val_dataset) * args.sample_ratio)
        val_indices = random.sample(range(len(val_dataset)), val_size)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
        
        test_size = int(len(test_dataset) * args.sample_ratio)
        test_indices = random.sample(range(len(test_dataset)), test_size)
        test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        print(f"采样后训练集: {len(train_dataset)}条 (原始: {original_train_size}条)")
        print(f"采样后验证集: {len(val_dataset)}条 (原始: {original_val_size}条)")
        print(f"采样后测试集: {len(test_dataset)}条 (原始: {original_test_size}条)")
    else:
        print(f"训练集: {len(train_dataset)}条")
        print(f"验证集: {len(val_dataset)}条")
        print(f"测试集: {len(test_dataset)}条")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_causal,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_causal,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_causal,
        num_workers=0
    )
    
    # 设置优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # 设置学习率调度器
    total_steps = len(train_loader) * args.epochs
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # 训练循环
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    print("\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            accumulation_steps=args.accumulation_steps
        )
        
        # 验证
        val_metrics, val_decoded = evaluate(model, val_loader, tokenizer, device)
        
        print(f"\n训练结果:")
        print(f"  Loss: {train_loss:.4f}")
        
        print(f"\n验证结果:")
        print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")
        
        # 记录历史
        train_history.append({'epoch': epoch + 1, 'loss': train_loss})
        val_history.append({
            'epoch': epoch + 1,
            **val_metrics
        })
        
        # 保存最优模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            
            # 保存模型
            if args.use_lora:
                model.save_pretrained(str(output_dir / 'lora_adapter'))
            else:
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
            
            print(f"OK 保存最优模型 (F1: {best_val_f1:.4f})")
    
    # 测试集评估
    print("\n" + "=" * 80)
    print("测试集评估...")
    test_metrics, test_decoded = evaluate(model, test_loader, tokenizer, device)
    
    print(f"\n测试结果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    
    # 保存测试结果
    results = {
        'test_metrics': test_metrics,
        'predictions': test_decoded
    }
    
    with open(output_dir / 'test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nOK 保存测试结果到 {output_dir / 'test_results.json'}")
    
    # 打印预测示例
    print("\n预测结果示例（前10条）:")
    print("-" * 80)
    for i, result in enumerate(test_decoded[:10]):
        status = "OK" if result['correct'] else "FAIL"
        print(f"{i+1}. 句子1: {result['sentence1'][:30]}...")
        print(f"   句子2: {result['sentence2'][:30]}...")
        print(f"   生成: {result['generated_text']} | 预测: {result['prediction']} | 实际: {result['label']} | {status}")
        print()
    
    # 保存训练历史
    history = {
        'train_history': train_history,
        'val_history': val_history,
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics
    }
    
    with open(output_dir / 'training_history.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"OK 保存训练历史到 {output_dir / 'training_history.json'}")
    
    # 保存配置信息
    with open(output_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)
    print(f"OK 保存配置信息到 {output_dir / 'config.json'}")
    
    # 生成详细报告
    generate_report(output_dir, history, test_decoded, args)
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print(f"最优验证F1: {best_val_f1:.4f}")
    print(f"测试集F1: {test_metrics['f1']:.4f}")
    print(f"所有结果保存在: {output_dir}")


def generate_report(output_dir, history, test_decoded, args):
    """生成详细报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BQ Corpus数据集 - Qwen文本匹配训练报告")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    report_lines.append("【配置参数】")
    report_lines.append("-" * 40)
    report_lines.append(f"模型路径: {args.model_path}")
    report_lines.append(f"最大序列长度: {args.max_len}")
    report_lines.append(f"批处理大小: {args.batch_size}")
    report_lines.append(f"训练轮数: {args.epochs}")
    report_lines.append(f"学习率: {args.learning_rate}")
    report_lines.append(f"数据采样比例: {args.sample_ratio}")
    report_lines.append(f"使用LoRA: {args.use_lora}")
    if args.use_lora:
        report_lines.append(f"LoRA r: {args.lora_r}")
        report_lines.append(f"LoRA alpha: {args.lora_alpha}")
    report_lines.append("")
    
    report_lines.append("【训练历史】")
    report_lines.append("-" * 40)
    report_lines.append(f"{'Epoch':<6} {'训练Loss':<12} {'验证Acc':<10} {'验证F1':<10}")
    report_lines.append("-" * 50)
    for train, val in zip(history['train_history'], history['val_history']):
        report_lines.append(f"{train['epoch']:<6} {train['loss']:<12.4f} {val['accuracy']:<10.4f} {val['f1']:<10.4f}")
    report_lines.append("")
    
    report_lines.append("【测试结果】")
    report_lines.append("-" * 40)
    metrics = history['test_metrics']
    report_lines.append(f"准确率: {metrics['accuracy']:.4f}")
    report_lines.append(f"精确率: {metrics['precision']:.4f}")
    report_lines.append(f"召回率: {metrics['recall']:.4f}")
    report_lines.append(f"F1分数: {metrics['f1']:.4f}")
    report_lines.append("")
    
    report_lines.append("【预测示例】")
    report_lines.append("-" * 40)
    for i, result in enumerate(test_decoded[:10]):
        status = "OK" if result['correct'] else "FAIL"
        report_lines.append(f"{i+1}. 句子1: {result['sentence1'][:40]}")
        report_lines.append(f"   句子2: {result['sentence2'][:40]}")
        report_lines.append(f"   生成: {result['generated_text']} | 预测: {result['prediction']} | 实际: {result['label']} | {status}")
        report_lines.append("")
    
    with open(output_dir / 'training_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"OK 保存训练报告到 {output_dir / 'training_report.txt'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BQ Corpus文本匹配训练 - 基于Qwen')
    
    # 数据参数
    parser.add_argument('--max_len', type=int, default=256, help='最大序列长度')
    parser.add_argument('--sample_ratio', type=float, default=0.1, help='数据集随机采样比例（0.0-1.0），默认0.1')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='pretrain_models/Qwen2___5-0___5B-Instruct', help='模型路径')
    
    # LoRA参数
    parser.add_argument('--use_lora', action='store_true', default=True, help='使用LoRA微调')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha参数')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积步数')
    
    args = parser.parse_args()
    
    main(args)
