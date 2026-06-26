"""
文本匹配数据集

"""

import json
import os

import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """就一个句对数据集，顺带把两种输入都吐出去。"""

    def __init__(self, data_path, tokenizer, max_length=64):
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                self.data.append((item['sentence1'], item['sentence2'], int(item['label'])))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence1, sentence2, label = self.data[idx]
        enc_pair = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        enc_a = self.tokenizer(
            sentence1,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )
        enc_b = self.tokenizer(
            sentence2,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

        item = {
            'input_ids': enc_pair['input_ids'].squeeze(0),
            'attention_mask': enc_pair['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),

            'input_ids_a': enc_a['input_ids'].squeeze(0),
            'attention_mask_a': enc_a['attention_mask'].squeeze(0),
            'input_ids_b': enc_b['input_ids'].squeeze(0),
            'attention_mask_b': enc_b['attention_mask'].squeeze(0),
        }

        if 'token_type_ids' in enc_pair:
            item['token_type_ids'] = enc_pair['token_type_ids'].squeeze(0)
        else:
            item['token_type_ids'] = torch.zeros_like(item['input_ids'])

        if 'token_type_ids' in enc_a:
            item['token_type_ids_a'] = enc_a['token_type_ids'].squeeze(0)
        else:
            item['token_type_ids_a'] = torch.zeros_like(item['input_ids_a'])

        if 'token_type_ids' in enc_b:
            item['token_type_ids_b'] = enc_b['token_type_ids'].squeeze(0)
        else:
            item['token_type_ids_b'] = torch.zeros_like(item['input_ids_b'])
        return item


def main():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data/lcqmc/train.jsonl')
    ds = PairDataset(data_path, tokenizer, max_length=64)

    print(f'Dataset size: {len(ds)}')
    print(f'\nFirst item keys: {ds[0].keys()}')
    print(f'\nShapes:')
    print(f'  input_ids:       {ds[0]["input_ids"].shape}')
    print(f'  attention_mask:  {ds[0]["attention_mask"].shape}')
    print(f'  token_type_ids:  {ds[0]["token_type_ids"].shape}')
    print(f'  input_ids_a:     {ds[0]["input_ids_a"].shape}')
    print(f'  input_ids_b:     {ds[0]["input_ids_b"].shape}')
    print(f'  label:           {ds[0]["label"]}')


if __name__ == '__main__':
    main()
