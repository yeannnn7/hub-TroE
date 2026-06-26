"""
检查超长文本样本
"""
from pathlib import Path
import json

DATA_DIR = Path(__file__).parent


def find_long_texts(dataset_name, threshold=100):
    """找出长度超过threshold的文本"""
    print(f"\n{'='*60}")
    print(f"检查 {dataset_name} 中的超长文本 (>{threshold}字符)")
    print(f"{'='*60}")

    splits = ['train', 'validation', 'test']
    total_long = 0

    for split in splits:
        file_path = DATA_DIR / dataset_name / f'{split}.jsonl'
        if not file_path.exists():
            continue

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        long_samples = []
        for idx, item in enumerate(data):
            len1 = len(item['sentence1'])
            len2 = len(item['sentence2'])
            if len1 > threshold or len2 > threshold:
                long_samples.append((idx, len1, len2, item))

        total_long += len(long_samples)
        print(f"\n{split} 集: 发现 {len(long_samples)} 条超长文本")

        # 显示前几个
        for idx, len1, len2, item in long_samples[:3]:
            print(f"\n  样本 #{idx}:")
            print(f"    sentence1 ({len1} chars): {repr(item['sentence1'][:100])}...")
            print(f"    sentence2 ({len2} chars): {repr(item['sentence2'][:100])}...")
            print(f"    label: {item['label']}")

    print(f"\n总共发现 {total_long} 条超长文本")
    return total_long


if __name__ == '__main__':
    find_long_texts('bq_corpus', threshold=100)
