# GraphRAG Tutorial | GraphRAG 教程

**基于知识图谱的 LLM 微调数据集生成与问答系统** | **Graph-based Dataset Generation & QA System for LLM Fine-tuning**

[English](#english-tutorial) | [中文](#中文教程)

---

# English Tutorial

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Document Preprocessing](#document-preprocessing)
5. [Dataset Generation (Core)](#dataset-generation-core)
6. [Building the RAG](#building-the-rag)
7. [Querying](#querying)
8. [Batch Processing & Validation](#batch-processing--validation)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## End-to-End Workflow

The typical workflow for generating a fine-tuning dataset:

```
1. preprocess   →  Convert raw PDFs/DOCX to clean Markdown
2. dataset      →  Generate Q&A pairs from Markdown (core step)
3. rag-build    →  Build knowledge graph for validation
4. batch_query  →  Validate dataset quality at scale
```

```bash
# Full pipeline example
python run.py preprocess --config graphrag/config/config.yaml
python run.py dataset --input-dir output/llm_compressed --output-dir output/dataset
python run.py rag-build
python batch_query.py --resume
```

---

## Prerequisites

Before you begin, ensure you have:

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Runtime |
| Neo4j | 5.x | Graph database (default: `bolt://localhost:7687`) |
| LLM API | OpenAI-compatible | Entity extraction & answer generation |
| Embedding API | OpenAI-compatible or local | Text embedding |

### Neo4j Setup

```bash
# Using Docker (recommended)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5-community

# Verify connection
cypher-shell -u neo4j -p your-password
```

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd GraphRag

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install preprocessing dependencies
pip install docling pypdf onnxruntime
```

---

## Configuration

All settings live in `graphrag/config/config.yaml`. Here's a minimal working configuration:

```yaml
# --- Embedding ---
embeddings:
  provider: openai
  model: text-embedding-v4
  openai_api_key: "sk-your-api-key"
  openai_base_url: "https://api.openai.com/v1"
  batch_size: 10

# --- LLM ---
llm:
  provider: openai
  model: gpt-4o
  api_key: "sk-your-api-key"
  base_url: "https://api.openai.com/v1"
  temperature: 0.3
  max_tokens: 2048

# --- Neo4j ---
neo4j:
  enabled: true
  uri: bolt://localhost:7687
  user: neo4j
  password: "your-password"
  database: neo4j

# --- Chunking ---
chunking:
  similarity_threshold: 0.75
  min_chunk_size: 50
  max_chunk_size: 2000

# --- Graph ---
graph:
  extraction_model_name: gpt-4o
  max_entities_per_chunk: 20
  max_relations_per_chunk: 40
  community_detection:
    resolution: 1.0

# --- Retrieval ---
retrieval:
  max_entities: 10
  neighbor_hops: 2
  max_community_results: 5

# --- Rerank (optional) ---
rerank:
  enabled: true
  provider: openai
  model: rerank-v3.5
  api_key: "sk-your-rerank-key"
  base_url: "https://api.rerank.com/v1"
  top_k: 5
  retrieve_k: 20
```

### Using Local Models

For offline/embedded deployment, use sentence-transformers:

```yaml
embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2

rerank:
  enabled: true
  provider: cross-encoder
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Document Preprocessing

The preprocessing pipeline converts raw documents (PDF, DOCX, etc.) into clean Markdown ready for RAG.

### Pipeline Stages

```
Raw Documents (PDF/DOCX)
    │
    ▼
Stage 1: Docling Conversion ──► Markdown files
    │
    ▼
Stage 2: Quality Filtering ──► Filtered Markdown
    │
    ▼
Stage 3: LLM Compression ──► Clean, compressed Markdown
```

### Running Preprocessing

```bash
# Run all stages
python run.py preprocess --config graphrag/config/config.yaml

# Run specific stage
python run.py preprocess --stage convert    # Stage 1 only
python run.py preprocess --stage quality    # Stage 2 only
python run.py preprocess --stage compress   # Stage 3 only

# With options
python run.py preprocess \
  --docs-dir documents/ \
  --device cuda \
  --quality-mode full \
  --min-quality 0.4 \
  --concurrency 4
```

### Preprocessing Options

| Option | Default | Description |
|--------|---------|-------------|
| `--docs-dir` | From config | Raw document directory |
| `--stage` | `all` | Which stage to run: `convert`, `quality`, `compress`, `all` |
| `--device` | `auto` | Inference device: `auto`, `cuda`, `cpu` |
| `--no-ocr` | `false` | Disable OCR |
| `--force-ocr` | `false` | Force OCR on all pages |
| `--quality-mode` | `full` | `stats`, `filter`, or `full` |
| `--min-quality` | `0.4` | Minimum quality score threshold |
| `--concurrency` | `1` | Parallel LLM compression workers |
| `--dry-run` | `false` | Estimate tokens without calling LLM |

---

## Building the RAG

### From Markdown Files

```bash
# Build from documents in the configured directory
python run.py rag-build -c graphrag/config/config.yaml

# Build from specific files
python run.py rag-build -f output/llm_compressed/doc1.md -f output/llm_compressed/doc2.md

# Build with graph export
python run.py rag-build --export output/graph.gexf --stats
```

### From Inline Text

```bash
python run.py rag-build \
  -d "doc1" "Apple is a technology company that makes iPhones." \
  -d "doc2" "Samsung makes Galaxy phones and semiconductors."
```

### What Happens During Build

1. **Document Loading** — Reads text files from specified paths
2. **Semantic Chunking** — Splits text at embedding similarity boundaries
3. **Entity Extraction** — LLM extracts entities and relationships per chunk
4. **Neo4j Storage** — Merges entities/relations into the graph database
5. **Community Detection** — Runs greedy modularity on the graph
6. **State Saving** — Pickles documents, chunks, and communities

### State Persistence

The build saves a state file (`output/rag_state.pkl`) containing:
- Documents (text content)
- Chunks (text + embeddings)
- Communities (entity groupings)

```bash
# Save to custom path
python run.py rag-build -o output/my_rag_state.pkl

# Load and reuse later
python run.py rag-query -s output/my_rag_state.pkl -q "Your question"
```

---

## Querying

### Single Query

```bash
python run.py rag-query -q "What are the safety requirements for pressure vessels?"
```

### Interactive Mode

```bash
python run.py rag-query -i

# Or just load without a query
python run.py rag-query

# Then type questions:
>>> What is the maximum operating temperature for Type 1 vessels?
>>> How often should safety valves be inspected?
>>> exit
```

### Query Pipeline

When you ask a question, GraphRAG:

1. **Extracts query terms** — Identifies keywords from your question
2. **Graph retrieval** — Finds matching entities, expands neighbors, retrieves communities, finds paths
3. **Chunk retrieval** — Embeds your question, finds top-k similar chunks
4. **Reranking** — (Optional) Re-ranks chunks for precision
5. **Generation** — Sends graph context + chunks to LLM for answer with citations

---

## Batch Processing & Validation

Process many questions at once to validate dataset quality at scale, with resume support.

### Setup

1. Create a JSONL dataset file (`output/dataset/dataset_all.jsonl`):

```jsonl
{"question": "What is the maximum pressure for Class 1 vessels?"}
{"question": "How often should safety valves be tested?"}
{"question": "What materials are approved for high-temperature service?"}
```

### Run Batch Query

```bash
# Process all questions
python batch_query.py

# Process with resume (skip already-answered questions)
python batch_query.py --resume

# Limit to first 100 questions
python batch_query.py --limit 100

# Custom paths
python batch_query.py \
  --dataset output/dataset/my_questions.jsonl \
  --output output/dataset/my_answers.jsonl \
  --state output/rag_state.pkl
```

### Batch Options

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `graphrag/config/config.yaml` | Config file path |
| `--dataset` | From config | Input JSONL dataset |
| `--output` | From config | Output JSONL file |
| `--state` | From config | RAG state file |
| `--limit` | `0` (all) | Max questions to process |
| `--resume` | From config | Skip already-answered questions |
| `--offset` | `0` | Skip first N questions |
| `--rebuild` | `false` | Force rebuild RAG (ignore saved state) |

### Output Format

Each record in the output JSONL includes:

```json
{
  "question": "What is the maximum pressure?",
  "answer": "According to GB/T 150, the maximum allowable pressure..."
}
```

---

## Dataset Generation (Core)

This is the core workflow — generating high-quality Q&A pairs for LLM fine-tuning from your documents.

```bash
# Generate from compressed Markdown
python run.py dataset --input-dir output/llm_compressed --output-dir output/dataset

# With options
python run.py dataset \
  --questions-per-chunk 5 \
  --max-chunk-tokens 24000 \
  --concurrency 4 \
  --no-chunk

# Dry run to estimate tokens and cost
python run.py dataset --dry-run

# Generate for specific files only
python run.py dataset --files 1-5
```

### How It Works

1. **Reads** compressed Markdown files from the input directory
2. **Chunks** long documents by section headers (or sends whole doc with `--no-chunk`)
3. **Sends** each chunk to the LLM with a specialized prompt that generates diverse, self-contained questions
4. **Deduplicates** and saves as JSONL (one question per line)
5. **Merges** all per-file JSONLs into `dataset_all.jsonl`

### Output Format

Each line in the output JSONL:

```json
{"question": "10kV配电变压器能效GBT32893 的附录C列举了哪几种变电站运行管理模式？", "source": "10kV配电变压器能效GBT32893.md"}
```

Questions are self-contained — they include the document identifier so they can be answered without additional context.

---

## Advanced Usage

### Python API

```python
from graphrag.src.main import GraphRAG

# Initialize
rag = GraphRAG(config_path="graphrag/config/config.yaml")

# Load documents
rag.load_documents({
    "safety标准1": "压力容器的设计压力应...",
    "safety标准2": "安全阀的校验周期...",
})

# Build
chunks = rag.chunk_documents(output_dir="output/chunks")
rag.build_graph()

# Query
answer = rag.query("压力容器的设计要求是什么？")
print(answer)

# Save state for later
rag.save("output/my_rag.pkl")

# Statistics
stats = rag.get_statistics()
print(f"Documents: {stats['documents']}, Chunks: {stats['chunks']}, "
      f"Nodes: {stats['graph_nodes']}, Edges: {stats['graph_edges']}")

# Export graph
rag.export_graph("output/knowledge_graph.gexf")

# Clean up
rag.close()
```

### Loading Saved State

```python
from graphrag.src.main import GraphRAG

# Load from saved state (no rebuild needed)
rag = GraphRAG.load("output/rag_state.pkl", config_path="graphrag/config/config.yaml")

# Query directly
answer = rag.query("Your question here")
rag.close()
```

### Custom Components

```python
from graphrag.src.embeddings.embedder import SentenceTransformerEmbedder
from graphrag.src.similarity.cosine import CosineSimilarity
from graphrag.src.main import GraphRAG
from graphrag.src.config.settings import load_config

# Use local embeddings
settings = load_config("graphrag/config/config.yaml")
settings.embeddings.provider = "sentence-transformers"
settings.embeddings.model = "all-MiniLM-L6-v2"

rag = GraphRAG(settings=settings)
```

### Knowledge Graph Export

Export the graph for visualization in Gephi:

```bash
# During build
python run.py rag-build --export output/graph.gexf

# The GEXF file includes:
# - Node attributes: type, description, community_id
# - Edge attributes: relation, description
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Neo4j connection failed` | Ensure Neo4j is running: `docker ps` or check `bolt://localhost:7687` |
| `OpenAI API error` | Check API key and base_url in config.yaml |
| `No documents loaded` | Verify `documents.directory` in config or use `-f` flag |
| `Out of memory` | Reduce `batch_size` in embeddings config, or use smaller model |
| `LLM extraction slow` | Reduce `max_entities_per_chunk`, increase `concurrency` |
| `Poor retrieval quality` | Adjust `similarity_threshold`, `neighbor_hops`, or `rerank.top_k` |

### Logs

Check logs in `logs/graphrag.log`:

```bash
tail -f logs/graphrag.log
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_graph.py -v

# Run with coverage
pytest tests/ --cov=graphrag --cov-report=html
```

---

# 中文教程

## 目录

1. [前置条件](#前置条件)
2. [安装](#安装)
3. [配置](#配置)
4. [文档预处理](#文档预处理)
5. [数据集生成（核心）](#数据集生成核心)
6. [构建 RAG](#构建-rag)
7. [查询](#查询)
8. [批量处理与验证](#批量处理与验证)
9. [高级用法](#高级用法)
10. [常见问题](#常见问题)

---

## 端到端工作流

生成微调数据集的典型流程：

```
1. preprocess   →  将原始 PDF/DOCX 转换为干净的 Markdown
2. dataset      →  从 Markdown 生成问答对（核心步骤）
3. rag-build    →  构建知识图谱用于验证
4. batch_query  →  大规模验证数据集质量
```

```bash
# 完整流水线示例
python run.py preprocess --config graphrag/config/config.yaml
python run.py dataset --input-dir output/llm_compressed --output-dir output/dataset
python run.py rag-build
python batch_query.py --resume
```

---

## 前置条件

开始之前，请确保已安装：

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | 3.10+ | 运行环境 |
| Neo4j | 5.x | 图数据库（默认：`bolt://localhost:7687`） |
| LLM API | OpenAI 兼容 | 实体提取与答案生成 |
| Embedding API | OpenAI 兼容或本地 | 文本向量化 |

### Neo4j 安装

```bash
# 使用 Docker（推荐）
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5-community

# 验证连接
cypher-shell -u neo4j -p your-password
```

---

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd GraphRag

# 2. 创建并激活虚拟环境
python -m venv .venv
source .venv/bin/activate

# 3. 安装核心依赖
pip install -r requirements.txt

# 4.（可选）安装预处理依赖
pip install docling pypdf onnxruntime
```

---

## 配置

所有设置在 `graphrag/config/config.yaml` 中。以下是最小可用配置：

```yaml
# --- 嵌入 ---
embeddings:
  provider: openai
  model: text-embedding-v4
  openai_api_key: "sk-your-api-key"
  openai_base_url: "https://api.openai.com/v1"
  batch_size: 10

# --- LLM ---
llm:
  provider: openai
  model: gpt-4o
  api_key: "sk-your-api-key"
  base_url: "https://api.openai.com/v1"
  temperature: 0.3
  max_tokens: 2048

# --- Neo4j ---
neo4j:
  enabled: true
  uri: bolt://localhost:7687
  user: neo4j
  password: "your-password"
  database: neo4j

# --- 切分 ---
chunking:
  similarity_threshold: 0.75
  min_chunk_size: 50
  max_chunk_size: 2000

# --- 图谱 ---
graph:
  extraction_model_name: gpt-4o
  max_entities_per_chunk: 20
  max_relations_per_chunk: 40
  community_detection:
    resolution: 1.0

# --- 检索 ---
retrieval:
  max_entities: 10
  neighbor_hops: 2
  max_community_results: 5

# --- 重排序（可选） ---
rerank:
  enabled: true
  provider: openai
  model: rerank-v3.5
  api_key: "sk-your-rerank-key"
  base_url: "https://api.rerank.com/v1"
  top_k: 5
  retrieve_k: 20
```

### 使用本地模型

离线/嵌入式部署可使用 sentence-transformers：

```yaml
embeddings:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2

rerank:
  enabled: true
  provider: cross-encoder
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## 文档预处理

预处理流水线将原始文档（PDF、DOCX 等）转换为干净的 Markdown，供 RAG 使用。

### 流水线阶段

```
原始文档 (PDF/DOCX)
    │
    ▼
阶段 1: Docling 转换 ──► Markdown 文件
    │
    ▼
阶段 2: 质量过滤 ──► 过滤后的 Markdown
    │
    ▼
阶段 3: LLM 压缩 ──► 干净、压缩的 Markdown
```

### 运行预处理

```bash
# 运行所有阶段
python run.py preprocess --config graphrag/config/config.yaml

# 运行特定阶段
python run.py preprocess --stage convert    # 仅阶段 1
python run.py preprocess --stage quality    # 仅阶段 2
python run.py preprocess --stage compress   # 仅阶段 3

# 带参数
python run.py preprocess \
  --docs-dir documents/ \
  --device cuda \
  --quality-mode full \
  --min-quality 0.4 \
  --concurrency 4
```

### 预处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--docs-dir` | 从配置读取 | 原始文档目录 |
| `--stage` | `all` | 运行阶段：`convert`、`quality`、`compress`、`all` |
| `--device` | `auto` | 推理设备：`auto`、`cuda`、`cpu` |
| `--no-ocr` | `false` | 关闭 OCR |
| `--force-ocr` | `false` | 强制对所有页面使用 OCR |
| `--quality-mode` | `full` | `stats`、`filter` 或 `full` |
| `--min-quality` | `0.4` | 最低质量分数阈值 |
| `--concurrency` | `1` | LLM 压缩并行数 |
| `--dry-run` | `false` | 仅估算 token，不调用 LLM |

---

## 构建 RAG

### 从 Markdown 文件构建

```bash
# 从配置目录中的文档构建
python run.py rag-build -c graphrag/config/config.yaml

# 从指定文件构建
python run.py rag-build -f output/llm_compressed/doc1.md -f output/llm_compressed/doc2.md

# 构建并导出图谱
python run.py rag-build --export output/graph.gexf --stats
```

### 从内联文本构建

```bash
python run.py rag-build \
  -d "doc1" "Apple 是一家科技公司，生产 iPhone。" \
  -d "doc2" "三星生产 Galaxy 手机和半导体。"
```

### 构建过程中发生了什么

1. **文档加载** — 从指定路径读取文本文件
2. **语义切分** — 在 embedding 相似度边界处切分文本
3. **实体提取** — LLM 从每个文本块中提取实体和关系
4. **Neo4j 存储** — 将实体/关系合并写入图数据库
5. **社区检测** — 在图上运行贪心模块度算法
6. **状态保存** — 序列化文档、文本块和社区信息

### 状态持久化

构建会保存状态文件（`output/rag_state.pkl`），包含：
- 文档（文本内容）
- 文本块（文本 + embedding）
- 社区（实体分组）

```bash
# 保存到自定义路径
python run.py rag-build -o output/my_rag_state.pkl

# 稍后加载复用
python run.py rag-query -s output/my_rag_state.pkl -q "你的问题"
```

---

## 查询

### 单次查询

```bash
python run.py rag-query -q "压力容器的安全要求是什么？"
```

### 交互模式

```bash
python run.py rag-query -i

# 或不带查询直接加载
python run.py rag-query

# 然后输入问题：
>>> I 型容器的最高工作温度是多少？
>>> 安全阀多久需要校验一次？
>>> exit
```

### 查询流程

当你提问时，GraphRAG 会：

1. **提取查询词** — 从问题中识别关键词
2. **图谱检索** — 查找匹配实体、扩展邻居、检索社区、查找路径
3. **文本块检索** — 嵌入问题，找到 top-k 相似文本块
4. **重排序** —（可选）重新排列文本块以提高精度
5. **生成** — 将图谱上下文 + 文本块发送给 LLM 生成带引用的答案

---

## 批量处理与验证

大规模处理问题以验证数据集质量，支持断点续传。

### 准备数据

创建 JSONL 数据集文件（`output/dataset/dataset_all.jsonl`）：

```jsonl
{"question": "I 型容器的最大许用压力是多少？"}
{"question": "安全阀多久需要测试一次？"}
{"question": "高温工况下允许使用哪些材料？"}
```

### 运行批量问答

```bash
# 处理所有问题
python batch_query.py

# 断点续传（跳过已有答案的问题）
python batch_query.py --resume

# 只处理前 100 条
python batch_query.py --limit 100

# 自定义路径
python batch_query.py \
  --dataset output/dataset/my_questions.jsonl \
  --output output/dataset/my_answers.jsonl \
  --state output/rag_state.pkl
```

### 批量参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `graphrag/config/config.yaml` | 配置文件路径 |
| `--dataset` | 从配置读取 | 输入 JSONL 数据集 |
| `--output` | 从配置读取 | 输出 JSONL 文件 |
| `--state` | 从配置读取 | RAG 状态文件 |
| `--limit` | `0`（全部） | 最大处理条数 |
| `--resume` | 从配置读取 | 跳过已有答案的问题 |
| `--offset` | `0` | 跳过前 N 条 |
| `--rebuild` | `false` | 强制重新构建 RAG |

### 输出格式

输出 JSONL 中每条记录包含：

```json
{
  "question": "最大许用压力是多少？",
  "answer": "根据 GB/T 150，最大许用压力为..."
}
```

---

## 数据集生成（核心）

这是核心工作流 — 从文档生成高质量问答对，用于 LLM 微调。

```bash
# 从压缩后的 Markdown 生成
python run.py dataset --input-dir output/llm_compressed --output-dir output/dataset

# 带参数
python run.py dataset \
  --questions-per-chunk 5 \
  --max-chunk-tokens 24000 \
  --concurrency 4 \
  --no-chunk

# 仅预估 token 用量和费用
python run.py dataset --dry-run

# 仅处理指定范围的文件
python run.py dataset --files 1-5
```

### 工作原理

1. **读取** 压缩后的 Markdown 文件
2. **分块** — 按章节标题分割长文档（或使用 `--no-chunk` 整篇发送）
3. **生成** — 将每个文本块发送给 LLM，使用专门的提示词生成多样化、自包含的问题
4. **去重** — 保存为 JSONL 格式（每行一个问题）
5. **合并** — 将所有单文件 JSONL 合并为 `dataset_all.jsonl`

### 输出格式

输出 JSONL 中每行格式：

```json
{"question": "10kV配电变压器能效GBT32893 的附录C列举了哪几种变电站运行管理模式？", "source": "10kV配电变压器能效GBT32893.md"}
```

问题都是自包含的 — 包含文档标识信息，无需额外上下文即可回答。

---

## 高级用法

### Python API

```python
from graphrag.src.main import GraphRAG

# 初始化
rag = GraphRAG(config_path="graphrag/config/config.yaml")

# 加载文档
rag.load_documents({
    "安全标准1": "压力容器的设计压力应...",
    "安全标准2": "安全阀的校验周期...",
})

# 构建
chunks = rag.chunk_documents(output_dir="output/chunks")
rag.build_graph()

# 查询
answer = rag.query("压力容器的设计要求是什么？")
print(answer)

# 保存状态供后续使用
rag.save("output/my_rag.pkl")

# 统计信息
stats = rag.get_statistics()
print(f"文档: {stats['documents']}, 块: {stats['chunks']}, "
      f"节点: {stats['graph_nodes']}, 边: {stats['graph_edges']}")

# 导出图谱
rag.export_graph("output/knowledge_graph.gexf")

# 清理
rag.close()
```

### 加载已保存状态

```python
from graphrag.src.main import GraphRAG

# 从保存的状态加载（无需重新构建）
rag = GraphRAG.load("output/rag_state.pkl", config_path="graphrag/config/config.yaml")

# 直接查询
answer = rag.query("你的问题")
rag.close()
```

### 自定义组件

```python
from graphrag.src.embeddings.embedder import SentenceTransformerEmbedder
from graphrag.src.similarity.cosine import CosineSimilarity
from graphrag.src.main import GraphRAG
from graphrag.src.config.settings import load_config

# 使用本地嵌入
settings = load_config("graphrag/config/config.yaml")
settings.embeddings.provider = "sentence-transformers"
settings.embeddings.model = "all-MiniLM-L6-v2"

rag = GraphRAG(settings=settings)
```

### 知识图谱导出

导出图谱用于 Gephi 可视化：

```bash
# 构建时导出
python run.py rag-build --export output/graph.gexf

# GEXF 文件包含：
# - 节点属性：type, description, community_id
# - 边属性：relation, description
```

---

## 常见问题

| 问题 | 解决方案 |
|------|----------|
| `Neo4j 连接失败` | 确保 Neo4j 正在运行：`docker ps` 或检查 `bolt://localhost:7687` |
| `OpenAI API 错误` | 检查 config.yaml 中的 API key 和 base_url |
| `未加载文档` | 验证配置中的 `documents.directory` 或使用 `-f` 参数 |
| `内存不足` | 减小 embeddings 配置中的 `batch_size`，或使用更小的模型 |
| `LLM 提取慢` | 减小 `max_entities_per_chunk`，增大 `concurrency` |
| `检索质量差` | 调整 `similarity_threshold`、`neighbor_hops` 或 `rerank.top_k` |

### 日志

查看日志文件 `logs/graphrag.log`：

```bash
tail -f logs/graphrag.log
```

### 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行指定测试文件
pytest tests/test_graph.py -v

# 带覆盖率
pytest tests/ --cov=graphrag --cov-report=html
```
