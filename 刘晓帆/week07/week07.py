import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from transformers import DataCollatorForTokenClassification
import numpy as np
from seqeval.metrics import classification_report, f1_score, accuracy_score
import pandas as pd
import json
import os
from typing import List, Dict, Tuple

# ==================== 1. 数据加载与预处理 ====================
class SequenceLabelingDataset(Dataset):
    """通用序列标注数据集类"""
    def __init__(self, texts, labels, tokenizer, label2id, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx]
        
        # Tokenization with word alignment
        tokenized_inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_overflowing_tokens=False,
            is_split_into_words=False,
        )
        
        # 如果有word-level labels，需要对齐到token-level
        if isinstance(text, list):  # 如果输入已经是分词列表
            words = text
            tokenized_inputs = self.tokenizer(
                words,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                is_split_into_words=True,
            )
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # 特殊token
                elif word_idx != previous_word_idx:
                    aligned_labels.append(self.label2id[word_labels[word_idx]])
                else:
                    aligned_labels.append(-100)  # 子词token
                previous_word_idx = word_idx
            tokenized_inputs["labels"] = aligned_labels
        else:
            # 对于整段文本，这里简化处理，需要根据实际标注格式调整
            tokenized_inputs["labels"] = [-100] * len(tokenized_inputs["input_ids"])
        
        return tokenized_inputs

# ==================== 2. 数据加载函数（支持多种格式）====================
def load_data_from_conll(file_path: str) -> Tuple[List, List]:
    """从CONLL格式加载数据"""
    texts = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        words = []
        word_labels = []
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                if words:
                    texts.append(words)
                    labels.append(word_labels)
                    words = []
                    word_labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    words.append(parts[0])
                    word_labels.append(parts[-1])  # 最后一列为标签
    
    if words:
        texts.append(words)
        labels.append(word_labels)
    
    return texts, labels

def load_data_from_json(file_path: str, text_key='text', label_key='labels') -> Tuple[List, List]:
    """从JSON格式加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [item[text_key] for item in data]
    labels = [item[label_key] for item in data]
    
    return texts, labels

def load_data_from_csv(file_path: str, text_col='text', label_col='labels') -> Tuple[List, List]:
    """从CSV格式加载数据"""
    df = pd.read_csv(file_path)
    texts = df[text_col].tolist()
    
    # 处理标签列（可能是字符串表示的列表）
    if isinstance(df[label_col].iloc[0], str):
        import ast
        labels = df[label_col].apply(ast.literal_eval).tolist()
    else:
        labels = df[label_col].tolist()
    
    return texts, labels

# ==================== 3. 模型训练配置 ====================
class SequenceLabelingTrainer:
    def __init__(self, 
                 model_name='bert-base-chinese',  # 或 'bert-base-uncased'
                 label_list=None,
                 max_length=128,
                 batch_size=16,
                 learning_rate=2e-5,
                 num_epochs=5,
                 output_dir='./output'):
        
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        
        # 标签映射
        self.label_list = label_list if label_list else ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # GPU支持
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def compute_metrics(self, p):
        """计算评估指标"""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        # 移除padding和特殊token
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            'accuracy': accuracy_score(true_labels, true_predictions),
            'f1': f1_score(true_labels, true_predictions),
            'report': classification_report(true_labels, true_predictions)
        }
    
    def train(self, train_texts, train_labels, eval_texts=None, eval_labels=None):
        """训练模型"""
        # 创建数据集
        train_dataset = SequenceLabelingDataset(
            train_texts, train_labels, self.tokenizer, self.label2id, self.max_length
        )
        
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = SequenceLabelingDataset(
                eval_texts, eval_labels, self.tokenizer, self.label2id, self.max_length
            )
        
        # 数据整理器
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            push_to_hub=False,
            logging_dir='./logs',
            logging_steps=10,
            report_to='none',
            metric_for_best_model='f1' if eval_dataset else None,
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if eval_dataset else None,
        )
        
        # 开始训练
        trainer.train()
        
        # 保存模型
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
    
    def predict(self, texts):
        """预测函数"""
        self.model.eval()
        
        # 确保输入是列表格式
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 解码预测结果
        results = []
        for i, text in enumerate(texts):
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
            pred_labels = [self.id2label[p.item()] for p in predictions[i]]
            
            # 过滤特殊token
            valid_pairs = [(token, label) for token, label in zip(tokens, pred_labels) 
                          if token not in ['[CLS]', '[SEP]', '[PAD]']]
            
            results.append(valid_pairs)
        
        return results

# ==================== 4. 主训练流程 ====================
def main():
    """
    主训练函数
    请根据您的数据集格式修改数据加载部分
    """
    
    # ========== 数据集配置 ==========
    # 请在这里指定您的数据集路径和格式
    # 示例1：CONLL格式
    # train_texts, train_labels = load_data_from_conll('path/to/train.conll')
    # eval_texts, eval_labels = load_data_from_conll('path/to/dev.conll')
    
    # 示例2：JSON格式
    # train_texts, train_labels = load_data_from_json('path/to/train.json')
    # eval_texts, eval_labels = load_data_from_json('path/to/dev.json')
    
    # 示例3：CSV格式
    # train_texts, train_labels = load_data_from_csv('path/to/train.csv')
    # eval_texts, eval_labels = load_data_from_csv('path/to/dev.csv')
    
    # 临时示例数据
    print("使用示例数据...")
    train_texts = [
        ["我", "爱", "北京", "天安门"],
        ["他", "在", "微软", "工作"],
    ]
    train_labels = [
        ["O", "O", "B-LOC", "I-LOC"],
        ["O", "O", "B-ORG", "O"],
    ]
    
    eval_texts = [
        ["上海", "是", "中国", "的", "城市"],
    ]
    eval_labels = [
        ["B-LOC", "O", "B-LOC", "O", "O"],
    ]
    
    # ========== 定义标签 ==========
    label_list = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
    
    # ========== 初始化训练器 ==========
    trainer = SequenceLabelingTrainer(
        model_name='bert-base-chinese',  # 中文用 bert-base-chinese，英文用 bert-base-uncased
        label_list=label_list,
        max_length=128,
        batch_size=8,
        learning_rate=2e-5,
        num_epochs=3,
        output_dir='./sequence_labeling_output'
    )
    
    # ========== 开始训练 ==========
    print("开始训练...")
    trained_model = trainer.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels
    )
    
    # ========== 测试预测 ==========
    print("\\n测试预测...")
    test_texts = ["我在北京工作"]
    predictions = trainer.predict(test_texts)
    
    for text, preds in zip(test_texts, predictions):
        print(f"\\n输入文本: {text}")
        print("预测结果:")
        for token, label in preds:
            print(f"  {token}: {label}")
    
    print(f"\\n训练完成！模型保存在: {trainer.output_dir}")

if __name__ == "__main__":
    main()
