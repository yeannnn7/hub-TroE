# 部署方式与效果对比

> 本文档汇总 `vllm_deployment` 项目中 **不同推理部署方式** 在吞吐、延迟、输出质量上的差异。  
> 数据来自 `outputs/throughput_results.json`、`outputs/function_call_results.json`。  
> 基座模型：**Qwen2-0.5B-Instruct**  
> **吞吐实测环境**：Mac **Darwin arm64 CPU**（`bench_throughput.py` 最新一次运行）  
> **Function Call 实测环境**：vLLM OpenAI Server（本地 CPU，`dtype=float32`）  
> **参考对比**：WSL2 + RTX 4060 Laptop 8GB（教学标准 GPU 环境，见 §2.2 对照表）

相关文档：[USAGE_GUIDE.md](./USAGE_GUIDE.md) | [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 1. 对比维度说明

本项目从两个正交维度做对比：

| 维度 | 问的问题 | 对应脚本 |
|------|----------|----------|
| **推理引擎 / 部署形态** | 怎么跑模型？吞吐多少？ | `bench_throughput.py`、`start_server.sh` |
| **解码约束方式** | 输出是否合法、能否过 Schema？ | `demo_guided_*.py`、`demo_function_call.py` |

二者可组合：例如 **vLLM HTTP Server + guided_json** 是生产 Agent 的常见形态。

---

## 2. 推理引擎与部署形态对比

### 2.1 五种方式一览

| 代号 | 方式 | 调用形态 | 典型场景 |
|------|------|----------|----------|
| **A** | Transformers 串行 | `model.generate()` 一次一条 | 本地调试、教学 baseline |
| **B** | Transformers 手动 batch | 手写 padding + `generate(batch=8)` | 简单批处理，无推理框架 |
| **C** | vLLM 离线引擎 | `LLM.generate()` 进程内调用 | 离线批推理、压测 |
| **D** | vLLM OpenAI Server | HTTP `POST /v1/chat/completions` | **生产部署主线**、与 OpenAI SDK 兼容 |
| **E** | 其他框架（扩展） | TGI / TensorRT-LLM / SGLang | 课后阅读，本项目未实测 |

**A/B** 不依赖 vLLM；**C/D** 依赖 vLLM。  
**D** 在 **C** 之上增加 FastAPI 服务层，单次请求多约 **1~5ms** HTTP 开销（相对 GPU 生成可忽略），但获得 **OpenAI 兼容 API + 约束解码参数**。

---

### 2.2 吞吐对比（50 条 prompt × 最多 100 new tokens）

测试条件（`bench_throughput.py`）：

- 50 条长短混合金融问答 prompt  
- `max_new_tokens=100`，`temperature=0`  
- Transformers batch 大小 = 8  

#### 实测数据（`outputs/throughput_results.json`，Mac Darwin arm64 CPU）

| 模式 | 总耗时 | QPS（请求/秒） | 生成 tok/s | 生成 token 数 | 相对 vLLM QPS |
|------|--------|----------------|------------|---------------|---------------|
| **[A] Transformers 串行** | 350.71 s | 0.14 | 10 | 3636 | **0.11×** |
| **[B] Transformers batch=8** | 366.33 s | 0.14 | 10 | 3636 | **0.11×** |
| **[C] vLLM continuous batching** | 38.61 s | **1.29** | **62** | 2412 | **1.00×** |

> Mac CPU 上 **batch=8 不比串行快**，甚至略慢（左 padding + CPU 矩阵乘开销）。GPU 上 batch 才有明显收益。

#### 参考数据（WSL2 + RTX 4060 8GB，历史 GPU 压测）

| 模式 | 总耗时 | QPS | tok/s | 相对 vLLM QPS |
|------|--------|-----|-------|---------------|
| A 串行 | 62.84 s | 0.80 | 58 | 0.018× |
| B batch=8 | 13.11 s | 3.81 | 283 | 0.087× |
| C vLLM | 1.15 s | 43.57 | 3043 | 1.00× |

> 不同机器、驱动、vLLM 小版本会导致 ±10% 波动；**同平台内的相对倍率**更有意义，**跨平台绝对值不可直接对比**。

#### 加速比

| 对比 | Mac CPU（最新） | WSL2 GPU（参考） |
|------|-----------------|------------------|
| vLLM vs 串行 | **~9×** QPS / **~6×** tok/s | **~54×** QPS / **~52×** tok/s |
| vLLM vs batch=8 | **~9.5×** QPS | **~11×** QPS |
| batch=8 vs 串行 | **~1.0×**（CPU 上无加速） | **~4.8×** QPS |

#### 为何 vLLM 快？

| 机制 | A/B 的问题 | vLLM 的做法 |
|------|------------|-------------|
| **KV cache 管理** | 固定长度 padding，内存碎片 | **PagedAttention**：按 block 分配 KV |
| **批调度** | 静态 batch，等最长序列 | **Continuous batching**：短请求结束即释放 slot |
| **GPU 利用率** | 串行时 GPU 大量空闲 | 动态 batch 可达 20~40 并发 slot |

**A → B** 在 **GPU** 上靠「凑 batch」就能快约 5 倍；在 **CPU** 上 batch 收益不明显。**B → C**（或 **A → C**）靠推理引擎 + continuous batching，Mac 上仍约有 **9×** QPS 提升，GPU 上可达 **11~54×**。

> 注意：本次 Mac 实测中 vLLM 生成 **2412** token，Transformers 生成 **3636** token（均设 `max_new_tokens=100`）。vLLM 更早触发 EOS/stop，tok/s 与绝对 token 数不宜与 A/B 逐条对齐，**QPS / 总墙钟时间**更可比。

---

### 2.3 延迟怎么理解？

吞吐 benchmark 测的是 **50 条请求的总墙钟时间**，不是单条 P99 延迟。可粗算 **平均每条墙钟时间**：

| 模式 | Mac CPU（最新） | WSL2 GPU（参考） | 说明 |
|------|-----------------|------------------|------|
| A 串行 | ~7.0 s/条 | ~1.26 s/条 | 严格串行，无排队 |
| B batch=8 | ~7.3 s/条 | ~0.26 s/条 | CPU 上 batch 无优势 |
| C vLLM | ~0.77 s/条 | ~0.023 s/条 | 批处理摊薄；单条单独发会更高 |

**注意**：vLLM 的 0.77 s/条（Mac）或 0.023 s/条（GPU）是 **高并发批处理下的平均值**。线上单用户首 token 延迟（TTFT）还需看 queue 深度、prefill 长度。

#### HTTP Server（D）相对离线引擎（C）

| 项 | 离线 `LLM.generate` | OpenAI Server |
|----|---------------------|---------------|
| 调用方式 | Python 进程内 | HTTP + JSON |
| 约束解码 | `SamplingParams(guided_decoding=...)` | `extra_body={"guided_json": ...}` |
| 额外开销 | 无 | 网络 + 序列化 ~1~5 ms |
| 适用 | 批处理脚本 | 微服务、Agent、多语言客户端 |

Function Call 实测（50 条经 Server，`outputs/function_call_results.json`）：单条平均 **0.43~0.60 s**（Mac CPU + float32），主要耗时在 **模型生成**，不是 HTTP。

---

### 2.4 资源与工程对比

| 项 | Transformers A/B | vLLM C/D |
|----|------------------|----------|
| 显存 / 内存（0.5B） | ~2 GB（GPU）/ 模型常驻内存（CPU） | GPU ~5 GB（`gpu_memory_utilization=0.6`）；CPU 用 float32 占更多 RAM |
| 约束解码 | ❌ 不支持 | ✅ guided_choice / regex / json |
| OpenAI 兼容 | ❌ | ✅（D） |
| 平台 | 全平台 | **Linux + CUDA 为主**（Mac 多为 CPU，可演示但不代表生产吞吐） |
| 上手成本 | 低 | 中（版本、驱动需对齐；Mac CPU 需 `float32`，见 §7） |

---

## 3. 解码约束方式对比（输出「效果」）

在 **vLLM OpenAI Server（D）** 上，同一模型、同一 prompt，换不同约束参数，**生成速度相近**，但 **可解析率 / Schema 通过率** 差异巨大。

### 3.1 四种约束级别

| 级别 | API 参数 | 保证什么 | 不保证什么 |
|------|----------|----------|------------|
| **裸 prompt** | 无 | 无 | JSON 语法、字段名、enum、正则 |
| **response_format** | `{"type":"json_object"}` | 输出是合法 JSON | 字段语义、enum 值、数值范围 |
| **guided_json** | `extra_body={"guided_json": schema}` | **完整 JSON Schema** | 业务语义是否正确 |
| **guided_choice / regex** | `guided_choice` / `guided_regex` | 枚举 / 正则格式 | 分类是否正确 |

原理：解码前构建 **FSM**，每步将非法 token 的 logit 置为 `-inf`，只能从合法集合采样。

---

### 3.2 Function Call 实测（`demo_function_call.py`，各 50 条）

数据来源：`outputs/function_call_results.json`

#### 工具 1：`get_stock_quote`（金融股价查询）

| 指标 | 裸 prompt | response_format | guided_json |
|------|-----------|-----------------|-------------|
| JSON 语法合法 | 43/50 (**86%**) | 50/50 (**100%**) | 50/50 (**100%**) |
| 必选字段齐全 | 43/50 (**86%**) | 50/50 (**100%**) | 50/50 (**100%**) |
| **完整 Schema 通过** | 30/50 (**60%**) | 34/50 (**68%**) | 50/50 (**100%**) |
| 50 条总延迟 | 21.83 s | 22.69 s | 21.71 s |
| **平均单条延迟** | **0.44 s** | **0.45 s** | **0.43 s** |

#### 工具 2：`create_order`（电商下单）

| 指标 | 裸 prompt | response_format | guided_json |
|------|-----------|-----------------|-------------|
| JSON 语法合法 | 48/50 (**96%**) | 50/50 (**100%**) | 50/50 (**100%**) |
| 必选字段齐全 | 48/50 (**96%**) | 50/50 (**100%**) | 50/50 (**100%**) |
| **完整 Schema 通过** | 21/50 (**42%**) | 21/50 (**42%**) | 50/50 (**100%**) |
| 50 条总延迟 | 28.87 s | 28.53 s | 29.87 s |
| **平均单条延迟** | **0.58 s** | **0.57 s** | **0.60 s** |

#### 核心结论（效果 vs 耗时）

1. **JSON 合法率**：`response_format` 与 `guided_json` 均可到 100%；裸 prompt 约 86%~96%。  
2. **Schema 通过率**：仅 **guided_json** 稳定 **100%**；`response_format` 对 enum/正则/范围 **几乎无帮助**（order 工具仍 42%）。  
3. **延迟**：三种模式单条平均 **0.43~0.60 s**，约束解码 **不显著增慢**（FSM 构建后可缓存）。  
4. **工程价值**：`response_format` 与 `guided_json` 在 Schema 通过率上可差 **30~58 个百分点**，Agent 下游解析失败率直接影响可用性。

#### 典型失败（裸 prompt / response_format 共有）

| 用户输入 | 失败原因 |
|----------|----------|
| 「300750 宁德时代最高价」 | `fields: ["最高价"]` 不在 enum `["open","close","high",...]` |
| 「订书…13711112222…支付宝」 | `user_phone: "+137..."` 违反手机号正则 |
| 「刷卡」 | `payment_method: "credit card"` 不在 enum `["alipay","wechat","card"]` |

`guided_json` 会在解码阶段 **屏蔽非法字符/枚举**，上述 case 在实测中 **0 失败**。

---

### 3.3 其他约束 demo（USAGE_GUIDE 预期）

| 脚本 | 约束类型 | 效果要点 | 延迟 |
|------|----------|----------|------|
| `demo_guided_choice.py` | 枚举 5 类意图 | 合法率 83% → **100%**；分类准确率看模型 | 与裸 prompt 同级 |
| `demo_guided_regex.py` | 正则 | 日期/股票代码格式一次根治 | 首次 FSM ~1~2 s，之后快 |
| `demo_guided_json.py` | Schema | 「22 年」→ 裸输出 `year:22`，guided 强制 ≥2015 | 同 function call |
| `demo_response_format.py` | OpenAI 标准 JSON | 跨厂商可移植，约束弱于 guided_json | 同 guided_json 量级 |

---

## 4. 综合选型矩阵

### 4.1 按业务目标选部署

| 目标 | 推荐方案 | 原因 |
|------|----------|------|
| 本地试 prompt | Transformers 串行（A） | 零依赖、最简单 |
| 离线洗数据 / 批标注 | vLLM 离线（C）或 batch（B） | 吞吐高 |
| **生产 API 服务** | **vLLM Server（D）** | OpenAI 兼容 + 约束解码 |
| Agent Function Call | D + **guided_json** | Schema 100% 可解析 |
| 意图路由 / 固定类别 | D + **guided_choice** | 输出必在枚举内 |
| 日期、编号等强格式 | D + **guided_regex** | 正则硬约束 |
| 多云部署、弱约束 JSON | D + **response_format** | 可移植，语法合法即可 |

### 4.2 按指标优先级选

| 最看重 | 首选 | 避免 |
|--------|------|------|
| **吞吐 / QPS** | vLLM（C/D） | Transformers 串行 |
| **单条延迟（低负载）** | vLLM Server + 小 batch | 串行 A |
| **输出可解析** | guided_json | 裸 prompt |
| **跨厂商兼容** | response_format | 仅 vLLM 私有参数 |
| **开发速度** | Transformers | 一上来全量 vLLM 运维 |

---

## 5. 一张图总结

```
                    输出质量（Schema 通过率）
                           ↑
                           │  guided_json / guided_choice / guided_regex
                           │  （vLLM Server + 约束解码）
                           │
         Transformers ─────┼───── vLLM Server (D)
         串行/batch (A/B)  │      ★ 生产推荐
                           │
                           └────────────────────────→ 吞吐 (QPS / tok/s)
                                    vLLM (C/D) >> batch (B) >> 串行 (A)
```

- **向右走**：换 vLLM、用 continuous batching → 吞吐 ↑↑（Mac CPU ~9×，GPU ~54×）  
- **向上走**：加 guided 约束 → 可解析率 ↑↑，延迟 ≈ 不变  

> GPU 上 **batch >> 串行**；Mac CPU 上 batch 几乎无收益，主要增益来自 **A/B → vLLM**。

---

## 6. 复现实验

### 6.1 吞吐对比

```bash
# 先停 server 释放显存
fuser -k 8000/tcp

cd src/
python bench_throughput.py
# → outputs/throughput_results.json
# → outputs/throughput_comparison.png
```

### 6.2 约束解码对比

```bash
bash start_server.sh   # 另开终端

python demo_function_call.py
# → outputs/function_call_results.json

python demo_guided_choice.py
python demo_guided_json.py
```

### 6.3 本地路径

`bench_throughput.py` 与 `start_server.sh` 已统一使用：

```python
MODEL_PATH = os.path.expanduser("~/Documents/www/py/llm/model/Qwen/Qwen2-0.5B-Instruct")
```

图表标题会自动显示本机设备（如 `Darwin arm64 CPU` 或 `NVIDIA GeForce RTX 4060 (8GB)`）。

---

## 7. 平台差异说明

| 环境 | 吞吐（50×100 token） | 约束解码 | 说明 |
|------|----------------------|----------|------|
| **Mac Darwin arm64 CPU** | A ~351 s / C ~39 s，vLLM **~9×** | ✅ 需 `float32` | **当前 `throughput_results.json` 来源**；见 `throughput_comparison.png` |
| WSL2 + RTX 4060 | A ~63 s / C ~1.2 s，vLLM **~54×** | ✅ 完整支持 | 教学标准 GPU 环境（§2.2 参考表） |
| Mac MPS | 未在本项目验证 | 视 vLLM 版本而定 | 生产仍推荐 Linux+CUDA |

**Mac CPU 约束解码报错**：`ValueError: logits must be of type float32`  
原因：vLLM 在 CPU 上默认 `float16`，而 xgrammar 的 CPU bitmask 内核只接受 `float32`。  
解决：`start_server.sh` 已自动检测 CPU 并改用 `float32`；或手动 `bash start_server.sh` 前设置 `DTYPE=float32`。

---

## 8. 相关产出文件

| 文件 | 内容 |
|------|------|
| `outputs/throughput_results.json` | A/B/C 吞吐原始数据 |
| `outputs/throughput_comparison.png` | 总耗时 / QPS / tok/s 柱状图（标题含本机设备名） |
| `outputs/function_call_results.json` | 双工具 × 三模式详细结果与失败样例 |

---

## 9. 总结

**Transformers 适合开发调试；vLLM 适合高吞吐生产推理（GPU 上相对串行可达 50×+，Mac CPU 演示约 9×）。在 vLLM Server 之上，`guided_json` 等约束解码用几乎相同的延迟（本地实测 0.43~0.60 s/条），把 Function Call 的 Schema 通过率从 42%~68% 拉到 100%——这是 Agent 系统能否稳定落地的关键差异。**
