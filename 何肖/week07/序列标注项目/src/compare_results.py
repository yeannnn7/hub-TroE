"""
汇总所有方案的评估结果，打印对比表

使用方式：
  python compare_results.py

前提（均在 作业/outputs/logs/ 下）：
  - eval_linear_validation.json   （python evaluate.py）
  - eval_crf_validation.json      （python evaluate.py --use_crf）
  - eval_llm.json                 （python llm_ner.py）
  - eval_sft.json                 （python evaluate_sft.py）

说明：
  - BERT 结果为 validation/test 全量 seqeval entity F1
  - LLM / SFT 为 validation 分层采样 100 条的 span F1（与 llm_ner.py 一致）
  - peoples_daily 的 test 集有标签，可用 evaluate.py --split test 另存结果
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "outputs" / "logs"


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # 读取各方案已落盘的 JSON 日志，汇总打印（不重新跑推理）
    linear_res = load_json(LOG_DIR / "eval_linear_validation.json")
    crf_res = load_json(LOG_DIR / "eval_crf_validation.json")
    llm_res = load_json(LOG_DIR / "eval_llm.json")
    sft_res = load_json(LOG_DIR / "eval_sft.json")

    print("\n" + "=" * 80)
    print("人民日报 NER 项目 — 五方案汇总对比")
    print("=" * 80)

    header = f"{'方案':<28} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 70)

    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + Linear':<28} "
            f"{linear_res['precision']:>10.4f} "
            f"{linear_res['recall']:>10.4f} "
            f"{linear_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + Linear':<28} {'（未找到，请运行 evaluate.py）':>50}")

    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(
            f"{'BERT + CRF':<28} "
            f"{crf_res['precision']:>10.4f} "
            f"{crf_res['recall']:>10.4f} "
            f"{crf_res['f1']:>10.4f} "
            f"{ill:>10d}"
        )
    else:
        print(f"{'BERT + CRF':<28} {'（未找到，请运行 evaluate.py --use_crf）':>50}")

    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "LLM")
        n = llm_res.get("n_samples", "?")
        print(
            f"{f'LLM zero-shot ({model_name})':<28} "
            f"{zs['precision']:>10.4f} "
            f"{zs['recall']:>10.4f} "
            f"{zs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(
            f"{f'LLM few-shot ({model_name})':<28} "
            f"{fs['precision']:>10.4f} "
            f"{fs['recall']:>10.4f} "
            f"{fs['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"\n  注：LLM 基于 validation 分层采样 {n} 条（span F1）")
    else:
        print(f"{'LLM zero/few-shot':<28} {'（未找到，请运行 llm_ner.py）':>50}")

    if sft_res:
        m = sft_res["metrics"]
        n = sft_res.get("n_samples", "?")
        print(
            f"{'Qwen2 SFT (LoRA)':<28} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} "
            f"{'N/A':>10}"
        )
        print(f"  注：SFT 基于 validation 分层采样 {n} 条（span F1）")
    else:
        print(f"{'Qwen2 SFT (LoRA)':<28} {'（未找到，请运行 evaluate_sft.py）':>50}")

    print("\n" + "=" * 80)
    print("关键结论：")
    # 以下跨方案 F1 对比为近似参考：BERT=全量 seqeval，LLM/SFT=100 条 span F1
    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"  2. 线性头非法序列：{ill_linear} 条；CRF：{crf_res['illegal_stats']['total_illegal']} 条")
    if llm_res and linear_res:
        fs_f1 = llm_res["few_shot"]["f1"]
        gap = linear_res["f1"] - fs_f1
        print(f"  3. BERT vs LLM few-shot：F1 差距 {gap:.4f}（指标口径略有不同，见文件头说明）")
    if sft_res and llm_res:
        sft_f1 = sft_res["metrics"]["f1"]
        fs_f1 = llm_res["few_shot"]["f1"]
        diff = sft_f1 - fs_f1
        print(f"  4. SFT vs LLM few-shot：F1 {'↑' if diff >= 0 else '↓'}{abs(diff):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
