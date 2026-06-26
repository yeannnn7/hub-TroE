"""
恢复原始数据脚本
功能：从备份中恢复原始数据
"""
from pathlib import Path
import shutil

DATA_DIR = Path(__file__).parent


def restore_dataset(dataset_name):
    """从备份恢复数据集"""
    backup_dir = DATA_DIR / f'{dataset_name}_backup'
    target_dir = DATA_DIR / dataset_name

    if not backup_dir.exists():
        print(f"✗ 未找到 {dataset_name} 的备份数据！")
        return False

    # 删除当前数据
    if target_dir.exists():
        shutil.rmtree(target_dir)

    # 从备份恢复
    shutil.copytree(backup_dir, target_dir)
    print(f"✓ {dataset_name} 已从备份恢复")
    return True


def main():
    datasets = ['bq_corpus', 'lcqmc']
    for name in datasets:
        restore_dataset(name)
    print("\n✓ 恢复完成！")


if __name__ == '__main__':
    main()
