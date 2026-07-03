"""
BQ Corpus数据集处理模块

用于Qwen大模型的文本匹配任务
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class BQCorpusDataset(Dataset):
    """BQ Corpus数据集"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        item = self.data[idx]
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = int(item['label'])
        
        # 构造输入文本
        # 使用对话格式：判断两个句子是否相似
        prompt = f"判断以下两个句子是否表达相同的含义：\n句子1：{sentence1}\n句子2：{sentence2}\n请回答：是或否"
        
        # 编码
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'sentence1': sentence1,
            'sentence2': sentence2
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """批处理函数"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'sentence1': [item['sentence1'] for item in batch],
        'sentence2': [item['sentence2'] for item in batch]
    }


class BQCorpusDatasetForCausal(Dataset):
    """BQ Corpus数据集 - 用于因果语言模型"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        item = self.data[idx]
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = int(item['label'])
        
        # 构造对话格式的输入
        # Qwen格式的对话模板
        answer = "是" if label == 1 else "否"
        
        prompt = f"<|im_start|>system\n你是一个文本匹配助手，判断两个句子是否表达相同的含义。<|im_end|>\n<|im_start|>user\n句子1：{sentence1}\n句子2：{sentence2}\n这两个句子是否表达相同的含义？<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        
        # 编码
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 对于因果语言模型，labels = input_ids
        labels = encoding['input_ids'].clone()
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0),
            'sentence1': sentence1,
            'sentence2': sentence2,
            'label': label
        }


def collate_fn_causal(batch: List[Dict]) -> Dict:
    """批处理函数 - 用于因果语言模型"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'sentence1': [item['sentence1'] for item in batch],
        'sentence2': [item['sentence2'] for item in batch],
        'label': [item['label'] for item in batch]
    }
