import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
from pathlib import Path

print("1. Starting Qwen2.5 LLM NER Training with LoRA...")

# ─────────────────── 默认路径（相对于 src/ 目录）────────────────────────────
ROOT          = Path(__file__).parent.parent
DATA_DIR      = ROOT / "data"/ "peoples_daily"
BERT_PATH     = ROOT / "pretrain_models" / "Qwen2___5-0___5B-Instruct"
OUTPUT_DIR    = ROOT / "outputs" / "llm"
CKPT_DIR      = OUTPUT_DIR / "checkpoints"
def parse_args():
    parser = argparse.ArgumentParser(description='Qwen2.5 NER Training with LoRA')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default=str(DATA_DIR))
    parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))

    # 模型参数
    parser.add_argument('--model_name_or_path', type=str, default=str(BERT_PATH))
    parser.add_argument('--max_len', type=int, default=64)
    
    # LoRA 参数
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=4, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,v_proj', 
                        help='Target modules for LoRA')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=50)
    
    # 数据缩放参数（用于缩小训练数据集，加速训练）
    parser.add_argument('--data_scale', type=int, default=20,
                        help='Data scale factor. Use 1 for full dataset, 20 for 1/20 of dataset')
    
    # 设备参数
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 training')
    
    return parser.parse_args()

class NERDataset(Dataset):
    """NER数据集类"""
    def __init__(self, data_path, tokenizer, max_len=128, label2id=None):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        if label2id is None:
            self.label2id = self._build_label_vocab()
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)
    
    def _load_data(self, data_path):
        """加载JSON数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_label_vocab(self):
        """构建标签词汇表"""
        labels = set(['O'])
        for item in self.data:
            for tag in item['ner_tags']:
                if tag not in labels:
                    labels.add(tag)
        return {label: idx for idx, label in enumerate(sorted(labels))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['ner_tags']
        
        # 分词处理
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True
        )
        
        # 对齐标签
        labels = []
        offset_mapping = encoded['offset_mapping']
        word_ids = encoded.word_ids()
        
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)  # 特殊token用-100忽略
            elif word_idx != previous_word_idx:
                if word_idx < len(tags):
                    labels.append(self.label2id.get(tags[word_idx], self.label2id['O']))
                else:
                    labels.append(self.label2id['O'])
                previous_word_idx = word_idx
            else:
                labels.append(-100)  # 子词用-100忽略
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'tokens': tokens,
            'original_tags': tags
        }

def compute_metrics(p):
    """计算评估指标"""
    predictions, labels = p
    
    # 获取预测标签（忽略-100）
    predictions = np.argmax(predictions, axis=2)
    
    # 过滤掉-100的标签
    mask = labels != -100
    predictions = predictions[mask]
    labels = labels[mask]
    
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def decode_entities(tokens, pred_labels, id2label):
    """解码实体"""
    entities = []
    current_entity = None
    
    for i, (token, label_id) in enumerate(zip(tokens, pred_labels)):
        label = id2label[label_id]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'text': token,
                'label': label[2:],
                'start': i,
                'end': i + 1
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
        elif label.startswith('E-') and current_entity:
            current_entity['text'] += token
            current_entity['end'] = i + 1
            entities.append(current_entity)
            current_entity = None
        elif label.startswith('S-'):
            entities.append({
                'text': token,
                'label': label[2:],
                'start': i,
                'end': i + 1
            })
        elif label == 'O':
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities

def evaluate_and_decode(model, dataloader, label2id, id2label, tokenizer, device):
    """评估并解码样本"""
    model.eval()
    decoded_samples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            if batch_idx < 5:
                # 使用predictions的batch size作为循环范围，避免索引越界
                batch_size = predictions.size(0)
                for i in range(batch_size):
                    tokens = batch['tokens'][i]
                    pred_labels = predictions[i].cpu().numpy()
                    
                    # 找到有效的预测标签（跳过特殊token）
                    word_ids = tokenizer(
                        tokens,
                        is_split_into_words=True,
                        max_length=args.max_len,
                        padding='max_length',
                        truncation=True
                    ).word_ids()
                    
                    valid_preds = []
                    prev_word_idx = None
                    for word_idx, pred in zip(word_ids, pred_labels):
                        if word_idx is not None and word_idx != prev_word_idx:
                            valid_preds.append(pred)
                            prev_word_idx = word_idx
                    
                    # 截断到原始token长度
                    valid_preds = valid_preds[:len(tokens)]
                    
                    entities = decode_entities(tokens, valid_preds, id2label)
                    decoded_samples.append({
                        'tokens': tokens,
                        'predicted_entities': entities,
                        'original_tags': batch['original_tags'][i]
                    })
    
    return decoded_samples

def main():
    global args
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Qwen2.5 NER Training Configuration")
    print("=" * 60)
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = NERDataset(
        os.path.join(args.data_dir, 'train.json'),
        tokenizer,
        max_len=args.max_len
    )
    valid_dataset = NERDataset(
        os.path.join(args.data_dir, 'validation.json'),
        tokenizer,
        max_len=args.max_len,
        label2id=train_dataset.label2id
    )
    test_dataset = NERDataset(
        os.path.join(args.data_dir, 'test.json'),
        tokenizer,
        max_len=args.max_len,
        label2id=train_dataset.label2id
    )
    
    # 保存标签信息（在数据缩放之前）
    num_labels = train_dataset.num_labels
    label2id = train_dataset.label2id
    id2label = train_dataset.id2label
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of labels: {num_labels}")
    print(f"Labels: {label2id}")
    
    # 数据缩放：按比例缩小训练数据集
    if args.data_scale > 1:
        original_size = len(train_dataset)
        new_size = len(train_dataset) // args.data_scale
        train_dataset = torch.utils.data.Subset(
            train_dataset, 
            list(range(new_size))  # 取前new_size个样本
        )
        print(f"\nData scaling applied: {original_size} -> {len(train_dataset)} samples (scale: {args.data_scale}x)")
    
    # 配置LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(','),
        lora_dropout=args.lora_dropout,
        bias='none',
        task_type='TOKEN_CLS',
    )
    
    # 加载模型
    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map='auto'
    )
    
    # 应用LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy='steps',
        save_strategy='steps',
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to='none',
        metric_for_best_model='f1',
        greater_is_better=True,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )
    
    # 开始训练
    print("\nStarting training...")
    trainer.train()
    
    # 保存LoRA权重
    print("\nSaving LoRA model...")
    model.save_pretrained(os.path.join(args.output_dir, 'lora_model'))
    
    # 在测试集上评估
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    print(f"Test Results: {test_results.metrics}")
    
    # 解码样本
    print("Decoding samples...")
    test_decoded = evaluate_and_decode(
        model, 
        DataLoader(test_dataset, batch_size=args.batch_size),
        label2id,
        id2label,
        tokenizer,
        args.device
    )
    
    # 保存结果
    results = {
        'configuration': vars(args),
        'label_mapping': {
            'label2id': label2id,
            'id2label': id2label
        },
        'training_results': trainer.state.log_history,
        'test_results': test_results.metrics,
        'test_decoded_samples': test_decoded
    }
    
    with open(os.path.join(args.output_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Test F1: {test_results.metrics.get('test_f1', 'N/A')}")
    print(f"Test Precision: {test_results.metrics.get('test_precision', 'N/A')}")
    print(f"Test Recall: {test_results.metrics.get('test_recall', 'N/A')}")
    print(f"\nAll results saved to: {args.output_dir}")
    print(f"LoRA model saved to: {os.path.join(args.output_dir, 'lora_model')}")
    print("=" * 60)

if __name__ == '__main__':
    main()
