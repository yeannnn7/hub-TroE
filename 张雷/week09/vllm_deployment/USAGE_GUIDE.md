# USAGE_GUIDE.md — 代码调用与测试指南

本文档覆盖从环境搭建到每个演示脚本的完整执行流程。支持 **macOS Metal（Apple Silicon）**和 **WSL2 Ubuntu + CUDA** 两种环境。

---

## 一、环境准备

### 1.1 macOS + Metal（Apple Silicon）

```bash
# 创建虚拟环境
python3 -m venv ~/.venv-vllm-metal
source ~/.venv-vllm-metal/bin/activate

# 安装 vLLM Metal 版
pip install vllm --index-url https://download.pytorch.org/whl/cpu   # 或按官方文档

# 安装依赖
cd ~/projects/llm-bootcamp/vllm_deployment
pip install -r requirements.txt matplotlib
```

**验证**：
```bash
source ~/.venv-vllm-metal/bin/activate
python -c "import vllm; print('vLLM:', vllm.__version__)"
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

### 1.2 WSL2 + Ubuntu 22.04 + CUDA（Windows 用户）

```powershell
# Windows 管理员 PowerShell
wsl --install -d Ubuntu-22.04
```

```bash
# Ubuntu 内
sudo apt update
sudo apt install -y python3-pip python3-venv build-essential git curl wget

python3 -m venv ~/vllm_env
source ~/vllm_env/bin/activate

mkdir -p ~/.pip
cat > ~/.pip/pip.conf << 'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

cd /mnt/d/badou/项目材料准备/vllm_deployment
pip install -r requirements.txt
```

### 1.3 关键兼容性

| 组件 | macOS Metal | WSL2 CUDA | 备注 |
|------|------------|-----------|------|
| vLLM | 0.24.0 (metal) | **0.9.2** | CUDA 版 0.20+ 需驱动 580+ |
| torch | 2.x + MPS | **2.7.0+cu126** | CUDA 版不可用 CUDA 13 |
| transformers | 4.x | **4.52.4** | 5.x 不兼容 vLLM 0.9.2 |
| 显存/统一内存 | 25.8GB (Metal) | 8GB (RTX 4060) | Mac 用统一内存 |

**macOS 注意事项**：
- PyTorch 编译时不带 CUDA，所有 `.to("cuda")` 需改为 `.to("mps")`
- `device_map="cuda"` → `device_map="mps"`
- `torch.cuda.empty_cache()` → `torch.mps.empty_cache()`

**WSL2 注意事项**：
- 如果 `torch.cuda.is_available()` 返回 False，降级 torch/vLLM 版本
- Windows 路径通过 `/mnt/d/...` 挂载，中文路径正常支持

---

## 二、启动 vLLM Server

所有 `demo_*.py` 脚本都通过 OpenAI 兼容 API 调用 server，必须先启动它。

### 2.1 启动

```bash
cd ~/projects/llm-bootcamp/vllm_deployment/src
bash start_server.sh
```

### 2.2 `start_server.sh` 关键配置说明

脚本中设置了以下环境变量（**macOS 必需**，解决 `198.18.0.1` 隧道接口问题）：

```bash
export MASTER_ADDR=127.0.0.1      # 强制 PyTorch distributed 走 loopback
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export GLOO_SOCKET_IFNAME=lo0     # gloo backend 绑定 lo0 而非 utun6
export TP_SOCKET_IFNAME=lo0       # tensor parallel 同样绑定 lo0
```

**问题背景**：macOS 的 utun6 隧道接口（VPN/私有网络）地址 `198.18.0.1` 被 PyTorch distributed 的 gloo backend 错误选用，导致 TCPStore 连接超时。WSL2 环境下这些变量非必需但无害。

### 2.3 验证可用

```bash
# 查询已加载模型
curl http://localhost:8000/v1/models

# 简单对话
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen2-0.5b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 50
  }'
```

### 2.4 停止 server

```bash
# macOS
lsof -ti:8000 | xargs kill -9

# WSL2 / Linux
fuser -k 8000/tcp
```

---

## 三、各脚本实测结果

以下为 **macOS Metal + Qwen2.5-0.5B-Instruct** 实测数据。

### 3.1 demo_guided_choice.py — 枚举约束

```bash
cd src/
python demo_guided_choice.py
```

**场景**：金融问答意图路由（查股价 / 查财报 / 查新闻 / 对比分析 / 其他），12 条测试用例。

**实测结果**：
```
指标                  裸 prompt           guided_choice
输出合法（在枚举内）     12/12 (100%)        12/12 (100%)
预测正确                5/12 (42%)         5/12 (42%)
平均延迟（秒）           0.459              0.311
```

**教学要点**：
- guided_choice 100% 保证输出合法（枚举约束在 token 级别生效）
- 分类正确率 42% 受限于 0.5B 模型的语义理解能力，约束解码不改模型本身
- guided_choice 略快（0.31s vs 0.46s），因 token 搜索空间被压缩到 5 个枚举值

**常见错误**：模型把"财报"和"股价"、"新闻"和"其他"搞混 — 这是模型能力问题，不是解码限制能解决的。

---

### 3.2 demo_guided_regex.py — 正则约束

```bash
python demo_guided_regex.py
```

**实测结果**：

任务 1（日期标准化 → YYYY-MM-DD）：
```
格式合法率：裸 prompt 3/6 (50%)  |  guided_regex 3/6 (50%)
```
模型对"三月三号"、"10月第一天"等需要推理的日期处理不当，正则只能约束格式，无法补偿推理。

任务 2（A 股代码抽取 → 6 位数字）：
```
格式合法率：裸 prompt 5/5 (100%)  |  guided_regex 5/5 (100%)
```
简单 6 位数字抽取两种方式都满分。

**教学要点**：regex 保证格式合法，但不补偿模型的推理/理解能力。

---

### 3.3 demo_guided_json.py — JSON Schema 基础

```bash
python demo_guided_json.py
```

**场景**：财报问答意图抽取（公司/年度/指标三元组），9 条测试。

**实测结果**：
```
指标                    裸 prompt        response_format   guided_json
合法 JSON              9/9 (100%)      9/9 (100%)       9/9 (100%)
字段齐全                9/9 (100%)      9/9 (100%)       9/9 (100%)
year 在 2015~2025      9/9 (100%)      9/9 (100%)       9/9 (100%)
metric 在枚举内         8/9 (89%)       8/9 (89%)        8/9 (89%)
jsonschema 完全通过     8/9 (89%)       8/9 (89%)        8/9 (89%)
```

**三种模式打成平手**。0.5B 对此简单 JSON 任务足够稳定，guided_json 的价值在复杂 schema（嵌套对象、数组、多字段约束）时才体现。

唯一失败：ICBC → 模型不认识是"工商银行"，输出 `metric: "营业收入"` 而非枚举内的 `"营收"`。

---

### 3.4 demo_response_format.py — OpenAI 标准方式

```bash
python demo_response_format.py
```

**场景**：新闻情感分类 + 置信度 + 关键词，5 条测试。

**实测结果**：裸 prompt 和 response_format **全部 5/5 (100%)**，0.5B 对简单 JSON 情感提取很稳定。

**教学要点**：`response_format={"type": "json_object"}` 是跨厂商可移植方案。当模型输出意愿强时和裸 prompt 无差别；当模型想输出解释文字时，它能强制纠正为 JSON。

---

### 3.5 demo_function_call.py ★ 核心

```bash
python demo_function_call.py          # 100 条全跑（约 35 分钟）
python demo_function_call.py --tool stock   # 只跑股票
python demo_function_call.py --tool order   # 只跑订单
```

**实测结果（get_stock_quote，50 条）**：
```
指标                    裸 prompt          response_format    guided_json
JSON 语法合法           50/50 (100%)      50/50 (100%)      50/50 (100%)
必选字段齐全             50/50 (100%)      50/50 (100%)      50/50 (100%)
完整 schema 通过 ★      46/50 ( 92%)      46/50 ( 92%)      46/50 ( 92%)
平均延迟（秒）            6.16              6.12              5.99
```

**实测结果（create_order，50 条）**：
```
指标                    裸 prompt          response_format    guided_json
JSON 语法合法           50/50 (100%)      50/50 (100%)      50/50 (100%)
必选字段齐全             50/50 (100%)      50/50 (100%)      50/50 (100%)
完整 schema 通过 ★      27/50 ( 54%)      27/50 ( 54%)      27/50 ( 54%)
平均延迟（秒）            7.78              7.83              7.71
```

**三种模式在 0.5B 模型上打成平手**。失败案例的共同特征：
- 股票代码拼错（`02699` 代替 `600000`）
- 手机号少一位（`1890001111` 代替 `18900001111`）
- priority 填错字段（`wechat` 放在 priority 而非 payment_method）
- 数量超限未修正（`200` > schema 最大值 100）

**核心结论**：
- guided_json 100% 保证**格式合法**（JSON 语法 + 字段齐全）
- 但**不能补偿模型推理能力**：当模型语义理解错了，schema 约束无法纠正
- 0.5B 级别模型上，三种模式的 schema 通过率等价；差距在更大模型（7B+）上会更明显
- `response_format` 只保证 JSON 语法，不保证字段值和类型 → 这是 guided_json 的工程价值所在

---

### 3.6 bench_throughput.py — 吞吐对比

```bash
# 先停 vLLM server（释放显存）
lsof -ti:8000 | xargs kill -9

source ~/.venv-vllm-metal/bin/activate
python bench_throughput.py
```

**实测结果（macOS Metal，50 prompts × 100 tokens）**：
```
模式                         总耗时      QPS       tokens/s    相对 vLLM
[A] transformers 串行        112.59s     0.44       44         0.16×
[B] transformers batch=8     128.91s     0.39       39         0.14×
[C] vLLM 批处理               17.98s     2.78      255         1.00×
```

**关键发现**：
- **vLLM 比 transformers 串行快 6.3 倍**，比 batch=8 快 7.2 倍
- **transformers batch=8 反而比串行慢**（128s vs 112s）— MPS 后端的典型特征，batch padding 额外开销 > 并行收益
- vLLM 在 Metal 上同样受益于 PagedAttention + continuous batching

**macOS 代码适配**（已将以下修改合入 `bench_throughput.py`）：
```python
# 原始（CUDA）→ 修改（MPS）
device_map="cuda"   → device_map="mps"
.to("cuda")         → .to("mps")
torch.cuda.empty_cache() → torch.mps.empty_cache()
```

**跑完后重新启动 server**：
```bash
bash start_server.sh
```

---

## 四、作为模块调用

### 4.1 启动 server 后用 OpenAI 客户端（推荐）

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# guided_json（vLLM 扩展）
resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "查茅台股价"}],
    extra_body={"guided_json": YOUR_SCHEMA},
    temperature=0,
)

# guided_choice（vLLM 扩展）
resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"guided_choice": ["选项A", "选项B", "选项C"]},
    temperature=0,
)

# 标准 response_format（OpenAI 兼容）
resp = client.chat.completions.create(
    model="qwen2-0.5b",
    messages=[{"role": "user", "content": "..."}],
    response_format={"type": "json_object"},
    temperature=0,
)
```

### 4.2 离线批处理（无 server）

```python
from vllm import LLM, SamplingParams

llm = LLM(model="/path/to/Qwen2-0.5B-Instruct",
          max_model_len=2048, gpu_memory_utilization=0.6,
          dtype="float16")

# 50 条批量推理，vLLM 自动 continuous batching
outputs = llm.generate(
    prompts,
    SamplingParams(temperature=0, max_tokens=100)
)
```

---

## 五、常见问题

### Q1：`ModuleNotFoundError: No module named 'vllm'`
先激活虚拟环境：`source ~/.venv-vllm-metal/bin/activate`（macOS）或 `source ~/vllm_env/bin/activate`（WSL2）。

### Q2：macOS 上 vLLM serve 启动后卡在 TCPStore 重试
```
[TCPStore.cpp:347] TCP client failed to connect/validate to host 198.18.0.1:52915
```
**原因**：PyTorch distributed 的 gloo backend 误用了 utun6 隧道接口。  
**解决**：设置环境变量强制走 loopback：
```bash
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0
export GLOO_SOCKET_IFNAME=lo0 TP_SOCKET_IFNAME=lo0
```
`start_server.sh` 已默认包含这些配置。

### Q3：`Torch not compiled with CUDA enabled`
macOS 上 PyTorch 不带 CUDA。所有 `.to("cuda")` → `.to("mps")`，`device_map="cuda"` → `device_map="mps"`。`bench_throughput.py` 已适配。

### Q4：`ModuleNotFoundError: No module named 'matplotlib'`
```bash
source ~/.venv-vllm-metal/bin/activate
pip install matplotlib
```

### Q5：`fuser -k 8000/tcp` 报 `Unknown option: k`
macOS 的 `fuser` 不支持 `-k`。替代命令：
```bash
lsof -ti:8000 | xargs kill -9
```

### Q6：demo_function_call.py 报 `The model 'Qwen2-0.5B-Instruct' does not exist`
模型名大小写不匹配。server 注册的是 `qwen2-0.5b`（全小写），脚本中 MODEL 变量需完全一致。已修复。

### Q7：demo 脚本三种模式结果完全相同
0.5B 模型上这是正常现象。小模型对简单 JSON 格式任务输出已经很稳定，guided_json 的优势（保证字段值合法）在复杂 schema 或更大模型上更明显。

### Q8：server 启动报 `ValueError: No available memory for the cache blocks`
显存/统一内存不足。降低 `gpu-memory-utilization`（0.6 → 0.4）或 `max-model-len`（2048 → 1024）。

### Q9：`aimv2 is already used by a Transformers config`
transformers 版本过新（5.x）。`pip install transformers==4.52.4`。

### Q10：Windows 路径在 WSL 里用 `/mnt/d/...` 正常吗
正常。WSL2 文件系统桥接层支持中文路径（UTF-8），模型权重只加载一次，速度影响可忽略。
