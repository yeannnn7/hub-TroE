# 日常出行规则问答助手（RAG）

基于检索增强生成（Retrieval-Augmented Generation）技术，针对中国日常出行相关法规与规章构建的智能问答系统。用户可用自然语言提问，系统从知识库中检索相关条款并由大语言模型生成准确回答。

提供**原生版**和**LangChain版**两种实现。

---

## 知识库内容

| 类别 | 文件 | 年份 |
|------|------|------|
| 公交 | 广州市公共汽车电车乘车守则 | 2024 |
| 地铁 | 广州城市轨道交通乘客守则 | 2024 |
| 地铁 | 深圳城市轨道交通运营管理办法 | - |
| 城市交通 | 城市公共交通条例 | 2024 |
| 民航 | 公共航空运输旅客服务管理规定 | 2021 |
| 民航 | 民用航空危险品运输管理规定 | 2024 |
| 民航 | 航班正常管理规定 | 2016 |
| 民航 | 锂电池航空运输规范 | - |
| 网约车 | 网络预约出租汽车运营服务规范 | - |
| 铁路 | 国铁集团铁路旅客运输规程 | 2024 |
| 铁路 | 禁止限制携带和托运物品目录 | 2022 |
| 铁路 | 铁路旅客运输规程（交通运输部） | 2023 |

---

## 项目结构

```
rag_annual_report/
├── data/
│   ├── raw_pdf/              # 原始 PDF 知识库文件
│   ├── parsed/               # 解析后的 JSON（自动生成）
│   ├── chunks/               # 分块结果（自动生成）
│   └── manifest.json         # 数据清单
├── vectorstore/              # 向量索引（自动生成）
├── models/                   # 本地 Embedding 模型
├── src/                      # 原生版
│   ├── parse_pdf.py          # PDF 解析
│   ├── chunk_documents.py    # 文档分块
│   ├── build_index.py        # 向量索引构建
│   ├── rag_pipeline.py       # RAG 问答核心
│   ├── serve.py              # FastAPI Web 服务
│   └── static/index.html     # 前端页面
├── src_langchain/            # LangChain 版
│   ├── build_index_lc.py     # 索引构建
│   ├── rag_chain_lc.py       # RAG 问答（LCEL 链）
│   └── download_model.py     # 下载本地 Embedding 模型
├── evaluation/               # 评测
│   ├── evaluate.py
│   ├── compare_strategies.py
│   └── questions.json        # 评测题集
└── requirements.txt
```

---

## 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 DashScope API Key（原生版 LLM + Embedding）
$env:DASHSCOPE_API_KEY = "sk-your-key-here"

# 3. LangChain 版需要下载本地 Embedding 模型
python src_langchain/download_model.py
```

---

## 快速启动

### 原生版

```bash
# 步骤 1：解析 PDF
python src/parse_pdf.py

# 步骤 2：分块
python src/chunk_documents.py

# 步骤 3：构建向量索引
python src/build_index.py

# 步骤 4a：终端交互问答
python src/rag_pipeline.py

# 步骤 4b：或启动 Web 服务
cd src
uvicorn serve:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

### LangChain 版

```bash
# 步骤 1：构建索引（含加载、分块、Embedding、建库）
python src_langchain/build_index_lc.py

# 步骤 2：终端交互问答
python src_langchain/rag_chain_lc.py

# 单次查询
python src_langchain/rag_chain_lc.py --query "地铁禁止携带哪些物品？"

# 附带来源文档
python src_langchain/rag_chain_lc.py --with-sources
```

---

## 两版对比

| 特性 | 原生版 | LangChain 版 |
|------|--------|-------------|
| 检索方式 | FAISS + BM25 混合检索 + RRF 融合 | FAISS 向量检索 |
| 查询改写 | 有（LLM 改写为检索友好的表述） | 无 |
| 元数据过滤 | 支持按出行类别过滤 | 无 |
| 分块策略 | semantic / fixed / hierarchical | RecursiveCharacterTextSplitter |
| LLM 调用 | OpenAI SDK 直接调用 | LangChain LCEL 链式调用 |
| Embedding | DashScope API（text-embedding-v3） | 本地 BAAI/bge-small-zh-v1.5 |
| Web 界面 | FastAPI + 前端 HTML | 无，仅终端 |
| 调试接口 | /query/debug 逐步展示中间结果 | 无 |
| 代码量 | ~430 行 | ~270 行 |
| 侧重点 | 教学演示，每步透明可控 | 工程实践，框架封装快速搭建 |

---

## 技术架构

```
用户提问
    │
    ▼
查询改写（原生版）──→ 改写后的问题
    │
    ├──────────────────────┐
    ▼                      ▼
FAISS 向量检索        BM25 关键词检索（原生版）
    │                      │
    └──────┬───────────────┘
           ▼
      RRF 融合排序（原生版）
           │
           ▼
      Top-K 文档片段
           │
           ▼
    LLM 生成回答 + 引用来源
```

---

## 评测

```bash
python evaluation/evaluate.py
python evaluation/compare_strategies.py
```

评测题集位于 `evaluation/questions.json`，涵盖公交、地铁、铁路、民航、网约车等出行类别的常见问题。

---

## 依赖说明

- **PDF 解析**：pdfplumber + PyMuPDF + pytesseract（OCR 备用）
- **向量检索**：faiss-cpu
- **关键词检索**：rank_bm25 + jieba
- **LLM**：阿里云 DashScope（qwen-plus），通过 OpenAI 兼容接口调用
- **Embedding**：原生版用 DashScope text-embedding-v3，LangChain 版用本地 bge-small-zh-v1.5
- **Web 服务**：FastAPI + Uvicorn
