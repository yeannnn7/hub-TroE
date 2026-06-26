# 分析文本匹配数据集
# 功能：统计正负例分布和文本长度分布，并生成可视化图表

from pathlib import Path
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ==================== 配置部分 ====================
# 设置matplotlib中文字体支持，确保图表可以正确显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# 数据集根目录（当前脚本所在目录）
DATA_DIR = Path(__file__).parent


def load_data(file_path):
    """
    加载jsonl格式的数据文件

    参数:
        file_path: 数据文件路径

    返回:
        包含所有样本的列表，每个样本是一个字典
        格式: [{"sentence1": "...", "sentence2": "...", "label": 0/1}, ...]
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data.append(json.loads(line))
    return data


def analyze_dataset(dataset_name):
    """
    分析单个数据集的统计信息

    参数:
        dataset_name: 数据集名称（'bq_corpus' 或 'lcqmc'）

    返回:
        包含训练集、验证集、测试集统计结果的字典
        结构: {
            'train': {'total': ..., 'positive': ..., 'negative': ..., 'length_stats': {...}, ...},
            'validation': {...},
            'test': {...}
        }
    """
    # 需要分析的数据集划分
    splits = ['train', 'validation', 'test']
    results = {}

    for split in splits:
        # 构建数据文件路径
        file_path = DATA_DIR / dataset_name / f'{split}.jsonl'
        if not file_path.exists():
            continue

        # 加载数据
        data = load_data(file_path)

        # ========== 统计正负例数量 ==========
        pos_count = sum(1 for item in data if item['label'] == 1)
        neg_count = sum(1 for item in data if item['label'] == 0)
        total_count = len(data)

        # ========== 统计文本长度 ==========
        s1_lengths = [len(item['sentence1']) for item in data]  # sentence1的长度列表
        s2_lengths = [len(item['sentence2']) for item in data]  # sentence2的长度列表
        all_lengths = s1_lengths + s2_lengths  # 所有文本的长度（s1和s2合并）

        # 计算长度统计指标
        results[split] = {
            'total': total_count,
            'positive': pos_count,
            'negative': neg_count,
            'pos_ratio': pos_count / total_count if total_count > 0 else 0,
            'neg_ratio': neg_count / total_count if total_count > 0 else 0,
            's1_lengths': s1_lengths,
            's2_lengths': s2_lengths,
            'all_lengths': all_lengths,
            'length_stats': {
                'min': np.min(all_lengths),       # 最小长度
                'max': np.max(all_lengths),       # 最大长度
                'mean': np.mean(all_lengths),     # 平均长度
                'median': np.median(all_lengths), # 中位数长度
                'p95': np.percentile(all_lengths, 95),  # 95分位数
                'p99': np.percentile(all_lengths, 99)   # 99分位数
            }
        }

    return results


def print_stats(dataset_name, results):
    """
    打印数据集统计信息到终端

    参数:
        dataset_name: 数据集名称
        results: analyze_dataset返回的统计结果
    """
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}")

    for split in ['train', 'validation', 'test']:
        if split not in results:
            continue
        res = results[split]
        print(f"\n--- {split} 集 ---")
        print(f"总样本数: {res['total']}")
        print(f"正例数: {res['positive']} ({res['pos_ratio']:.2%})")
        print(f"反例数: {res['negative']} ({res['neg_ratio']:.2%})")
        print(f"\n长度统计:")
        print(f"  最小长度: {res['length_stats']['min']}")
        print(f"  最大长度: {res['length_stats']['max']}")
        print(f"  平均长度: {res['length_stats']['mean']:.2f}")
        print(f"  中位数长度: {res['length_stats']['median']}")
        print(f"  P95长度: {res['length_stats']['p95']}")
        print(f"  P99长度: {res['length_stats']['p99']}")


def plot_length_distribution(dataset_name, results, output_dir):
    """
    绘制文本长度分布直方图

    参数:
        dataset_name: 数据集名称
        results: analyze_dataset返回的统计结果
        output_dir: 图表保存目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)  # 如果目录不存在则创建

    # 创建3行1列的子图（训练集、验证集、测试集各一个）
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f'{dataset_name} 文本长度分布', fontsize=16)

    splits = ['train', 'validation', 'test']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿三种颜色

    for idx, split in enumerate(splits):
        if split not in results:
            continue
        ax = axes[idx]
        res = results[split]

        # 绘制直方图，x轴范围限制在p99+5，避免异常长文本影响可视化效果
        all_lengths = res['all_lengths']
        max_show = int(res['length_stats']['p99']) + 5
        ax.hist(all_lengths, bins=50, range=(0, max_show), alpha=0.7, color=colors[idx], label=split)

        # 在图表右上角添加统计信息标注
        stats = res['length_stats']
        label_text = (f"min={stats['min']}, max={stats['max']}\n"
                      f"mean={stats['mean']:.1f}, median={stats['median']}\n"
                      f"p95={stats['p95']:.0f}, p99={stats['p99']:.0f}")
        ax.text(0.95, 0.95, label_text, transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 设置图表标题和坐标轴标签
        ax.set_title(f'{split} 集 (样本数: {res["total"]})')
        ax.set_xlabel('文本长度')
        ax.set_ylabel('样本数量')
        ax.grid(True, alpha=0.3)  # 添加网格线
        ax.legend()

    plt.tight_layout()  # 自动调整子图间距
    plt.savefig(output_dir / f'{dataset_name}_length_distribution.png', dpi=150)  # 保存为高分辨率PNG
    plt.close()  # 关闭图表释放内存


def plot_label_distribution(dataset_name, results, output_dir):
    """
    绘制正负例分布柱状图

    参数:
        dataset_name: 数据集名称
        results: analyze_dataset返回的统计结果
        output_dir: 图表保存目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # 获取存在的数据集划分
    splits = [s for s in ['train', 'validation', 'test'] if s in results]
    x = np.arange(len(splits))  # x轴位置
    width = 0.35  # 柱子宽度

    # 提取正负例数量
    pos_counts = [results[s]['positive'] for s in splits]
    neg_counts = [results[s]['negative'] for s in splits]

    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pos_counts, width, label='正例', color='#2ca02c')  # 正例用绿色
    rects2 = ax.bar(x + width/2, neg_counts, width, label='反例', color='#d62728')  # 反例用红色

    # 设置图表标题和坐标轴标签
    ax.set_xlabel('数据集划分')
    ax.set_ylabel('样本数量')
    ax.set_title(f'{dataset_name} 正负例分布')
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.legend()

    # 在柱子顶部添加数值标签的辅助函数
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width()/2., height),
                        xytext=(0, 3),  # 向上偏移3个像素
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(output_dir / f'{dataset_name}_label_distribution.png', dpi=150)
    plt.close()


def main():
    """
    主函数：依次分析所有数据集
    """
    # 要分析的数据集列表
    datasets = ['bq_corpus', 'lcqmc']
    # 图表输出目录
    output_dir = DATA_DIR / 'figures'

    for dataset_name in datasets:
        # 1. 分析数据集
        results = analyze_dataset(dataset_name)
        # 2. 打印统计信息
        print_stats(dataset_name, results)
        # 3. 绘制长度分布图
        plot_length_distribution(dataset_name, results, output_dir)
        # 4. 绘制正负例分布图
        plot_label_distribution(dataset_name, results, output_dir)
        print(f"\n图表已保存到 {output_dir}")


if __name__ == '__main__':
    main()
