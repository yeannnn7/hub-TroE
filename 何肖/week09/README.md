# vLLM 部署与约束解码实验

基于 **Qwen2-0.5B-Instruct**，使用 vLLM 0.9.2 部署 OpenAI 兼容推理服务，对比 **裸 prompt / response_format / guided_json** 等约束解码方案，验证结构化输出在 Function Calling 场景下的可靠性。

## 环境

| 组件 | 版本 |
|------|------|
| vLLM | 0.9.2 |
| GPU | Tesla V100-PCIE-32GB |
| CUDA | True |
| 模型 | Qwen2-0.5B-Instruct |

```bash
source ~/vllm_env/bin/activate
python -c "import vllm, torch; print('vLLM:', vllm.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 项目结构

```
vllm_deployment/
├── src/
│   ├── start_server.sh          # 启动 vLLM OpenAI 兼容服务
│   ├── demo_guided_choice.py    # 枚举约束
│   ├── demo_guided_regex.py     # 正则约束
│   ├── demo_guided_json.py      # JSON Schema 约束
│   ├── demo_response_format.py  # OpenAI JSON mode
│   ├── demo_function_call.py    # ★ Function Calling 综合对比
│   └── bench_throughput.py      # 吞吐 benchmark
├── outputs/
│   ├── function_call_results.json   # demo_function_call 详细结果
│   └── throughput_results.json      # bench_throughput 吞吐对比
├── homework.md                  # 完整运行记录
└── USAGE_GUIDE.md               # 详细使用说明
```

## 运行流程

```bash
# 1. 启动 vLLM Server（另开终端）
cd src && bash start_server.sh

# 2. 约束解码 Demo
python demo_guided_choice.py
python demo_guided_regex.py
python demo_guided_json.py
python demo_response_format.py

# 3. 核心：Function Calling 对比（100 用例，结果写入 outputs/）
python demo_function_call.py

# 4. 吞吐 benchmark（结果写入 outputs/）
python bench_throughput.py
```

## 实验结果汇总

### guided_choice（意图分类，12 条）

| 指标 | 裸 prompt | guided_choice |
|------|-----------|---------------|
| 输出合法（在枚举内） | 12/12 (100%) | 12/12 (100%) |
| 预测正确 | 5/12 (42%) | 5/12 (42%) |
| 平均延迟 | 0.155s | 0.699s |

**结论**：guided_choice 保证输出必在枚举内；本实验中分类准确率与裸 prompt 相同，但避免了非法 token。

---

### guided_regex（日期 / 股票代码）

| 任务 | 裸 prompt 格式合法率 | guided_regex |
|------|---------------------|--------------|
| 日期 → YYYY-MM-DD | 3/6 (50%) | **6/6 (100%)** |
| A 股代码 → 6 位数字 | 5/5 (100%) | 5/5 (100%) |

**结论**：guided_regex 保证下游解析器始终拿到合法格式（日期、代码等严格字段）。

---

### guided_json vs response_format（9 条金融查询）

| 指标 | 裸 prompt | response_format | guided_json |
|------|-----------|-----------------|-------------|
| 合法 JSON | 9/9 | 9/9 | 9/9 |
| jsonschema 完全通过 | 8/9 (89%) | 8/9 (89%) | **9/9 (100%)** |

**结论**：response_format 只保证是 JSON；guided_json 才能保证字段类型、枚举、正则全部符合 schema。

---

### response_format（情感分析，5 条）

| 指标 | 裸 prompt | response_format |
|------|-----------|-----------------|
| 合法 JSON | 5/5 | 5/5 |
| 字段齐全 & 值合法 | 5/5 | 5/5 |

**结论**：简单 JSON 结构下两者表现接近；复杂 schema 仍需 guided_json。

---

### demo_function_call.py ★（核心，100 用例）

> 数据来源：`outputs/function_call_results.json`

#### get_stock_quote（50 条）

| 指标 | 裸 prompt | response_format | guided_json |
|------|-----------|-----------------|-------------|
| JSON 语法合法 | 50/50 | 50/50 | 50/50 |
| 必选字段齐全 | 50/50 | 50/50 | 50/50 |
| 完整 schema 通过 | 46/50 (92%) | 46/50 (92%) | **50/50 (100%)** |
| 总耗时 | 35.8s | 58.0s | 207.6s |
| 平均延迟 | 0.716s | 1.159s | 4.153s |

`guided_json` 无失败案例。`raw` / `response_format` 典型失败（见 outputs）：
- symbol `02699` 不符合 `^\d{6}$`
- 多余字段 `symbols`、`isBuy`

#### create_order（50 条）

| 指标 | 裸 prompt | response_format | guided_json |
|------|-----------|-----------------|-------------|
| JSON 语法合法 | 50/50 | 50/50 | 50/50 |
| 必选字段齐全 | 50/50 | 50/50 | 50/50 |
| 完整 schema 通过 | 28/50 (56%) | 28/50 (56%) | **46/50 (92%)** |
| 总耗时 | 44.6s | 66.6s | 288.1s |
| 平均延迟 | 0.891s | 1.331s | 5.761s |

`guided_json` 剩余 4 条失败均为 **quantity 超范围**（0、200、1000 等），属输入本身违反 schema 约束。

---

### bench_throughput.py（吞吐对比）

> 数据来源：`outputs/throughput_results.json`  
> 配置：50 prompts · max_new_tokens=100 · batch_size=8 · Qwen2-0.5B

| 模式 | 耗时 | 生成 tokens | QPS | TPS (tok/s) |
|------|------|-------------|-----|-------------|
| Transformers 串行 | 151.1s | 5000 | 0.33 | 33.1 |
| Transformers batch=8 | 24.5s | 5000 | 2.04 | 204.2 |
| **vLLM** | **1.9s** | 4634 | **25.77** | **2388.1** |

**结论**：vLLM 相对串行 TPS 提升约 **72×**，相对手写 batch 提升约 **12×**，体现 PagedAttention + continuous batching 收益。

---

## 主要结论

1. **裸 prompt**：JSON 语法虽高，但 schema 合规率低（create_order 仅 56%），生产不可用。
2. **response_format**：JSON 合法率提升，但 schema 通过率与裸 prompt 相同（stock 92%、order 56%）。
3. **guided_json**：复杂 schema 下最可靠（stock 100%、order 92%）；代价是延迟最高（约 4–6×）。
4. **vLLM 吞吐**：相对 Transformers 串行 TPS **72×**，是部署层的基础收益。
5. **工程选型**：简单 JSON → `response_format`；Function Calling / 严格 API 参数 → `guided_json`。

## 输出文件

| 文件 | 生成脚本 | 内容 |
|------|----------|------|
| `outputs/function_call_results.json` | `demo_function_call.py` | 100 用例 × 3 模式：schema 通过数、总延迟、失败案例（user/output/error） |
| `outputs/throughput_results.json` | `bench_throughput.py` | serial / batch / vLLM 的 QPS、TPS、耗时 |

`function_call_results.json` 结构示例：

```json
{
  "stock": { "n": 50, "stats": { "raw": {...}, "guided_json": {...} }, "fails": {...} },
  "order": { "n": 50, "stats": {...}, "fails": {...} }
}
```

---

## 附录：完整运行记录

详见 [homework.md](./homework.md)
