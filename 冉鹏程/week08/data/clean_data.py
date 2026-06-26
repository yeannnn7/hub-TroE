"""
数据清洗脚本
功能：清洗bq_corpus和lcqmc数据集中的异常长文本
"""
from pathlib import Path
import json
import shutil

DATA_DIR = Path(__file__).parent


def clean_text(text, max_len=128):
    """
    清洗单个文本：尝试修复被拼接的文本，然后截断到合理长度

    参数:
        text: 原始文本
        max_len: 最大允许长度

    返回:
        清洗后的文本
    """
    # 1. 尝试修复被\t拼接的文本
    if '\t' in text:
        # 如果有制表符，说明是多个样本被拼接到一起了，取第一部分
        text = text.split('\t')[0]

    # 2. 截断到最大长度
    text = text[:max_len]

    return text


def clean_dataset(dataset_name, max_len=128, min_len=1, remove_abnormal=True, backup=True):
    """
    清洗整个数据集

    参数:
        dataset_name: 数据集名称 ('bq_corpus' 或 'lcqmc')
        max_len: 最大文本长度，超过此长度会被截断
        min_len: 最小文本长度，短于此长度的样本会被删除
        remove_abnormal: 是否删除超长异常样本（可选）
        backup: 是否备份原始数据
    """
    source_dir = DATA_DIR / dataset_name

    if backup:
        # 备份原始数据
        backup_dir = DATA_DIR / f'{dataset_name}_backup'
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(source_dir, backup_dir)
        print(f"✓ 原始数据已备份到: {backup_dir}")

    splits = ['train', 'validation', 'test']
    total_removed = 0
    total_truncated = 0

    for split in splits:
        file_path = source_dir / f'{split}.jsonl'
        if not file_path.exists():
            continue

        # 读取原始数据
        with open(file_path, 'r', encoding='utf-8') as f:
            original_data = [json.loads(line.strip()) for line in f if line.strip()]

        cleaned_data = []
        removed_count = 0
        truncated_count = 0

        for item in original_data:
            s1 = item['sentence1']
            s2 = item['sentence2']

            # 检查是否需要截断
            s1_truncated = clean_text(s1, max_len)
            s2_truncated = clean_text(s2, max_len)

            if s1_truncated != s1 or s2_truncated != s2:
                truncated_count += 1

            # 检查是否需要删除
            if len(s1_truncated) < min_len or len(s2_truncated) < min_len:
                removed_count += 1
                continue

            # 可选：如果原文本异常长且包含制表符，直接删除（更保守的策略）
            if remove_abnormal and ('\t' in s1 or '\t' in s2):
                removed_count += 1
                continue

            cleaned_data.append({
                'sentence1': s1_truncated,
                'sentence2': s2_truncated,
                'label': item['label']
            })

        # 写回清洗后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in cleaned_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        total_removed += removed_count
        total_truncated += truncated_count

        print(f"\n{split} 集:")
        print(f"  原始样本数: {len(original_data)}")
        print(f"  清洗后样本数: {len(cleaned_data)}")
        print(f"  删除样本数: {removed_count}")
        print(f"  截断样本数: {truncated_count}")

    print(f"\n{'='*60}")
    print(f"{dataset_name} 清洗完成！")
    print(f"  总共删除: {total_removed} 条")
    print(f"  总共截断: {total_truncated} 条")
    print(f"{'='*60}")


def main():
    """
    主函数：清洗两个数据集
    """
    print("开始清洗数据...")

    # 清洗 bq_corpus - 这个数据集有异常
    print("\n" + "="*60)
    print("清洗 bq_corpus (有异常长文本)")
    print("="*60)
    clean_dataset(
        'bq_corpus',
        max_len=128,      # 最大保留128字符
        min_len=1,        # 最少1个字符
        remove_abnormal=True,  # 删除被拼接的异常样本
        backup=True       # 备份原始数据
    )

    # 清洗 lcqmc - 这个数据质量较好，只做基本处理
    print("\n" + "="*60)
    print("清洗 lcqmc (数据质量较好)")
    print("="*60)
    clean_dataset(
        'lcqmc',
        max_len=128,
        min_len=1,
        remove_abnormal=False,  # 不需要删除
        backup=True
    )

    print("\n✓ 全部清洗完成！")


if __name__ == '__main__':
    main()
