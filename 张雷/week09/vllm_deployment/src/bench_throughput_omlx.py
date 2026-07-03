"""
吞吐对比：oMLX 串行 / 并发 pool=4 / 并发 pool=8

教学重点：
  1. oMLX 是 Apple Silicon 原生的 LLM 推理服务器，类似 vLLM
  2. 内部实现了 continuous batching: 不同长度请求动态组 batch
  3. 同一个模型、同一批请求，并发越高吞吐越大

测试方法：
  50 个长短混合的问答 prompt（从短到长），目标生成 100 token
  三路分别测总耗时、QPS（请求/秒）、token/s（生成速度）
  产出柱状图到 outputs/throughput_omlx.png

环境：
  Mac M3 24GB + oMLX (brew) + Qwen2-0.5B-Instruct
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

# ── 配置 ──────────────────────────────────────────────────────────────
OMLX_BASE = "http://localhost:8000/v1"
API_KEY = "omlx-eusvga3p6a73wlk4"
MODEL_NAME = "Qwen2-0.5B-Instruct"
N_PROMPTS = 50
MAX_NEW_TOKENS = 100
POOL_SMALL = 4
POOL_LARGE = 8   # 对齐 oMLX 默认 max_concurrent_requests

# ── 测试 prompts（长短混合，模拟真实业务）────────────────────────────
SHORT_QUESTIONS = [
    "什么是股票？", "什么是基金？", "什么是ETF？", "什么是债券？", "什么是期权？",
    "什么是熊市？", "什么是牛市？", "什么是PE？", "什么是ROE？", "什么是毛利率？",
]
MEDIUM_QUESTIONS = [
    "解释一下价值投资和趋势投资的区别。",
    "什么情况下应该止损？",
    "为什么会出现股市崩盘？",
    "沪深300和中证500有什么区别？",
    "什么是量化交易？",
    "基金定投的优势是什么？",
    "股票回购对股价有什么影响？",
    "可转债有哪些特点？",
    "如何判断一家公司是否值得投资？",
    "什么是做市商制度？",
]
LONG_QUESTIONS = [
    "请详细介绍一下巴菲特的投资理念及其核心原则，并举例说明。",
    "解释下现金流折现（DCF）估值法的基本步骤、使用的参数以及它的局限性。",
    "比较A股和美股在交易制度、监管环境、投资者结构等方面的主要差异。",
    "什么是技术分析？它和基本面分析有什么区别？两种方法各自的适用场景是什么？",
    "详细解释资产配置的核心思想，常见的几种配置模型，以及如何根据个人风险偏好调整。",
]
PROMPTS = (SHORT_QUESTIONS * 3 + MEDIUM_QUESTIONS * 1 + LONG_QUESTIONS * 2)[:N_PROMPTS]
assert len(PROMPTS) == N_PROMPTS


# ══════════════════════════════════════════════════════════════════════
#                      HTTP 请求工具
# ══════════════════════════════════════════════════════════════════════

def make_request(prompt: str) -> int:
    """发送单条 chat completion 请求，返回生成的 token 数"""
    resp = requests.post(
        f"{OMLX_BASE}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_NEW_TOKENS,
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["usage"]["completion_tokens"]


# ══════════════════════════════════════════════════════════════════════
#                     三档测试
# ══════════════════════════════════════════════════════════════════════

def bench_serial(prompts: list[str]) -> dict:
    """[A] 串行：逐条发送，等一条完成再发下一条"""
    print("\n[A] oMLX 串行（逐条发送）...")
    total_tokens = 0
    t0 = time.time()
    for i, p in enumerate(prompts):
        total_tokens += make_request(p)
        if (i + 1) % 10 == 0:
            print(f"    进度 {i+1}/{len(prompts)}")
    dt = time.time() - t0
    return {
        "time": dt, "gen_tokens": total_tokens,
        "qps": len(prompts) / dt, "tps": total_tokens / dt,
    }


def bench_pool(prompts: list[str], max_workers: int) -> dict:
    """并发请求，控制最大并发数"""
    label = f"pool={max_workers}"
    print(f"\n     oMLX 并发（ThreadPoolExecutor {label}）...")
    total_tokens = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(make_request, p): i for i, p in enumerate(prompts)}
        for i, fut in enumerate(as_completed(futures)):
            total_tokens += fut.result()
            if (i + 1) % 10 == 0:
                print(f"    完成 {i+1}/{len(prompts)}")
    dt = time.time() - t0
    return {
        "time": dt, "gen_tokens": total_tokens,
        "qps": len(prompts) / dt, "tps": total_tokens / dt,
    }


# ══════════════════════════════════════════════════════════════════════
#                     绘图 + 报告
# ══════════════════════════════════════════════════════════════════════

def plot_results(r: dict, out_path: str):
    modes = [
        "oMLX\nserial",
        f"oMLX\npool={POOL_SMALL}",
        f"oMLX\npool={POOL_LARGE}\n(continuous batching)",
    ]
    keys = ["serial", "pool_small", "pool_large"]
    times = [r[k]["time"] for k in keys]
    qps = [r[k]["qps"] for k in keys]
    tps = [r[k]["tps"] for k in keys]
    colors = ["#aab7c4", "#82b1ff", "#69f0ae"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.rcParams["axes.unicode_minus"] = False

    # 1. 总耗时
    bars = axes[0].bar(modes, times, color=colors)
    axes[0].set_ylabel("Time (seconds)")
    axes[0].set_title(f"Total Time for {N_PROMPTS} Requests")
    for b, v in zip(bars, times):
        axes[0].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}s",
                     ha="center", va="bottom")

    # 2. QPS
    bars = axes[1].bar(modes, qps, color=colors)
    axes[1].set_ylabel("QPS (requests/sec)")
    axes[1].set_title("Requests Per Second (higher is better)")
    for b, v in zip(bars, qps):
        axes[1].text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                     ha="center", va="bottom")

    # 3. tokens/s
    bars = axes[2].bar(modes, tps, color=colors)
    axes[2].set_ylabel("Tokens / sec (generated)")
    axes[2].set_title("Generation Throughput (tokens/sec)")
    for b, v in zip(bars, tps):
        axes[2].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                     ha="center", va="bottom")

    plt.suptitle("oMLX Throughput Benchmark (Qwen2-0.5B, Mac M3 24GB)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n柱状图已保存：{out_path}")


def main():
    print("=" * 70)
    print(f"  oMLX Throughput Benchmark  |  {N_PROMPTS} prompts × max {MAX_NEW_TOKENS} new tokens")
    print(f"  模型: {MODEL_NAME}  |  Mac M3 24GB")
    print("=" * 70)

    # 先快速验证连接
    print("\n验证 oMLX 连接...")
    try:
        make_request("Hello")
        print("  oMLX API 连接正常 ✓")
    except Exception as e:
        print(f"  ✗ 连接失败: {e}")
        return

    # 三档测试
    results = {}
    results["serial"] = bench_serial(PROMPTS)
    print(f"\n[B] 并发对比")
    results["pool_small"] = bench_pool(PROMPTS, max_workers=POOL_SMALL)
    results["pool_large"] = bench_pool(PROMPTS, max_workers=POOL_LARGE)

    # ── 汇总表 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  结果汇总")
    print("=" * 70)
    print(f"{'模式':<35}{'总耗时':<12}{'QPS':<10}{'tokens/s':<12}{'相对串行':<10}")
    print("-" * 80)
    speedup_base = results["serial"]["qps"]
    name_map = {
        "serial":     "[A] oMLX 串行",
        "pool_small": f"[B] oMLX pool={POOL_SMALL}",
        "pool_large": f"[C] oMLX pool={POOL_LARGE}",
    }
    for k in ["serial", "pool_small", "pool_large"]:
        r = results[k]
        rel = r["qps"] / speedup_base
        print(f"{name_map[k]:<33}{r['time']:>6.2f}s     "
              f"{r['qps']:>5.2f}     {r['tps']:>6.0f}      {rel:>5.2f}×")

    # ── 保存结果 ────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "throughput_omlx_results.json")
    png_path = os.path.join(out_dir, "throughput_omlx.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_prompts": N_PROMPTS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "model": MODEL_NAME,
            "backend": "oMLX (Apple Silicon)",
            "pool_small": POOL_SMALL,
            "pool_large": POOL_LARGE,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 结果保存：{json_path}")

    plot_results(results, png_path)

    print("\n" + "=" * 70)
    print("  核心结论：")
    print(f"    oMLX pool={POOL_LARGE} 相对串行加速：{results['pool_large']['qps']/results['serial']['qps']:.1f}×")
    print(f"    oMLX pool={POOL_SMALL} 相对串行加速：{results['pool_small']['qps']/results['serial']['qps']:.1f}×")
    print("    关键机制：oMLX 内置 continuous batching (PagedAttention on Metal)")
    print("=" * 70)


if __name__ == "__main__":
    main()
