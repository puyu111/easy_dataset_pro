"""调用远端 LLM 对 Markdown 文档进行智能压缩与质量提升。

功能:
  1. 读取 Markdown 文件
  2. 发送到远端 LLM (OpenAI 兼容 API)
  3. LLM 对文档进行压缩/清洗/结构化优化
  4. 输出更高质量的压缩版 .md 文件
  5. 对比原始/压缩文件，输出统计数据

用法:
    python -m graphrag.src.preprocessing.llm_compress                                          # 全部压缩
    python -m graphrag.src.preprocessing.llm_compress --files 2-5                              # 指定范围
    python -m graphrag.src.preprocessing.llm_compress --dry-run                                # 预估 token
    python -m graphrag.src.preprocessing.llm_compress --max-chunk-tokens 32000                 # 每段最大 token
    python -m graphrag.src.preprocessing.llm_compress --concurrency 3                          # 并发数
    python -m graphrag.src.preprocessing.llm_compress --compare                                # 对比已有结果

配置:
  从 .env 文件或环境变量读取 LLM 配置:
    OPENAI_API_KEY=sk-xxx
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL=deepseek-v4-flash
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# ── load .env ────────────────────────────────────────────────────────────────
def load_env(env_path: str = None):
    """Load environment variables from .env file."""
    if env_path is None:
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    env_path = Path(env_path)
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                os.environ.setdefault(key, value)


load_env()

from openai import OpenAI

# ── 默认路径 ─────────────────────────────────────────────────────────────────
# 通过环境变量或函数参数覆盖
_DEFAULT_INPUT_DIR = Path(os.environ.get(
    "LLM_COMPRESS_INPUT_DIR",
    str(Path.cwd() / "output" / "filtered_md"),
))
_DEFAULT_OUTPUT_DIR = Path(os.environ.get(
    "LLM_COMPRESS_OUTPUT_DIR",
    str(Path.cwd() / "output" / "llm_compressed"),
))
_DEFAULT_STATE_FILE = Path(os.environ.get(
    "LLM_COMPRESS_STATE_FILE",
    str(Path.cwd() / "output" / "llm_compress_state.json"),
))
_DEFAULT_REPORT_FILE = Path(os.environ.get(
    "LLM_COMPRESS_REPORT_FILE",
    str(Path.cwd() / "output" / "llm_compress_report.json"),
))


# ── LLM 客户端 ───────────────────────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")


# ── 压缩提示词 ───────────────────────────────────────────────────────────────
COMPRESS_SYSTEM_PROMPT = """你是一名专业的技术文档编辑，擅长中文安全标准/规范类文档的清洗与压缩。

## 任务

对用户提供的原始 Markdown 文档进行智能压缩与质量提升，输出更高品质的 Markdown。

## 要求

1. **内容保真**：保留所有技术参数、数值、分类、公式、标准编号、日期等关键信息。不遗漏、不改写、不概括技术细节。

2. **格式清洗**：
   - 移除 PDF 转换产生的噪声（如 "I C S" 中插空格、"/G21/G22" 类乱码标记、<!-- 图片缺失 -->注释）
   - 修复编码问题，如 Yi 音节字符（犐、犆、犛 等 → 对应正常汉字）
   - 删除多余的空格、空行、页面分隔符（--- pagebreak ---）
   - 统一中英文标点符号

3. **结构优化**：
   - 确保标题层级合理（# → ## → ###）
   - 表格、列表格式规范整洁
   - 条款/条文保持清晰编号
   - 如有目录，保留目录结构

4. **压缩目标**：
   - 去除冗余表达、重复内容
   - 合并相近条款
   - 精简非必要的修饰性语言
   - 输出应比原文更紧凑、更易读

## 输出格式

直接输出优化后的 Markdown 内容，不要包含任何解释、说明或额外标记。
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文 ~1.5 token/字，英文 ~0.25 token/字符）。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + ascii_chars * 0.25)


def calculate_cost(input_tokens: int, output_estimate: int) -> float:
    """估算 DeepSeek V4 Flash 费用 (元)。"""
    input_cost = input_tokens / 1_000_000 * 1.0
    output_cost = output_estimate / 1_000_000 * 2.0
    return input_cost + output_cost


def read_md(filepath: Path) -> str:
    """读取 Markdown 文件，处理可能的编码问题。"""
    for enc in ["utf-8", "gbk", "gb18030", "latin-1"]:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, max_tokens: int = 24000) -> list[str]:
    """将长文本按章节分割为不超过 max_tokens 的块。"""
    sections = re.split(r'(?=^##\s)', text, flags=re.MULTILINE)
    if len(sections) <= 1:
        sections = re.split(r'(?=^#\s)', text, flags=re.MULTILINE)

    chunks = []
    current_chunk = ""
    for section in sections:
        section_tokens = estimate_tokens(section)
        current_tokens = estimate_tokens(current_chunk)

        if current_tokens + section_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = section
        else:
            current_chunk += "\n" + section if current_chunk else section

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks if chunks else [text]


def merge_chunks(chunks: list[str]) -> str:
    """合并多个压缩块为一个完整文档。"""
    return "\n\n".join(chunks)


def compress_with_llm(
    text: str,
    max_tokens: int = 8192,
    retries: int = 3,
    client: OpenAI | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
) -> str:
    """发送文本到 LLM 进行压缩。"""
    if not text.strip():
        return ""

    _client = client or OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    _model = model or LLM_MODEL
    _system = system_prompt or COMPRESS_SYSTEM_PROMPT

    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": _system},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            result = resp.choices[0].message.content.strip()
            if result.startswith("```"):
                result = re.sub(r'^```\w*\n?', '', result)
                result = re.sub(r'\n?```$', '', result)
            return result
        except Exception as e:
            err_msg = str(e)
            # 余额不足、认证失败等不可恢复错误直接抛出
            if "402" in err_msg or "Insufficient Balance" in err_msg or "401" in err_msg:
                raise
            if attempt < retries - 1:
                wait = 2 ** attempt
                logger.warning("  重试 %d/%d (%ds): %s", attempt + 1, retries, wait, e)
                time.sleep(wait)
            else:
                raise

    return ""


def process_file(
    filepath: Path,
    output_dir: Path,
    max_chunk_tokens: int,
    max_output_tokens: int,
    state: dict,
    lock: Lock,
    client: OpenAI | None = None,
    model: str | None = None,
    no_chunk: bool = False,
) -> dict:
    """处理单个文件：读取 -> (可选分块) -> LLM 压缩 -> 保存。

    Args:
        no_chunk: True 时不分割文本，整篇文档一次发给 LLM。
    """
    filename = filepath.name
    output_path = output_dir / filename

    with lock:
        if state.get(filename, {}).get("status") == "completed":
            return {"file": filename, "status": "skipped", "reason": "already completed"}

    # 输出文件已存在则跳过（防止 state 文件丢失时重复压缩）
    if output_path.exists() and output_path.stat().st_size > 100:
        with lock:
            if filename not in state or state.get(filename, {}).get("status") != "completed":
                state[filename] = {"status": "completed", "output_tokens": estimate_tokens(read_md(output_path)), "chunks": 1, "compression_ratio": 0}
        return {"file": filename, "status": "skipped", "reason": "output already exists"}

    try:
        text = read_md(filepath)
        input_tokens = estimate_tokens(text)

        if len(text.strip()) < 50:
            with lock:
                state[filename] = {"status": "skipped", "reason": "too short"}
            return {"file": filename, "status": "skipped", "reason": "too short"}

        _client = client or OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        _model = model or LLM_MODEL

        if no_chunk:
            # 不分块：整篇文档一次发送；超出上下文时自动回退分块
            try:
                compressed = compress_with_llm(text, max_tokens=max_output_tokens, client=_client, model=_model)
                output_tokens = estimate_tokens(compressed)
                chunks_used = 1
            except Exception as e:
                err_msg = str(e)
                if "maximum context length" in err_msg or "reduce the length" in err_msg:
                    logger.warning("  %s: 超出上下文限制，自动回退分块模式", filename)
                    chunks = chunk_text(text, max_tokens=max_chunk_tokens)
                    if len(chunks) == 1:
                        compressed = compress_with_llm(chunks[0], max_tokens=max_output_tokens, client=_client, model=_model)
                        output_tokens = estimate_tokens(compressed)
                    else:
                        compressed_chunks = []
                        for i, chunk in enumerate(chunks):
                            logger.info("  分段 %d/%d (%d tokens)", i + 1, len(chunks), estimate_tokens(chunk))
                            compressed_chunk = compress_with_llm(chunk, max_tokens=max_output_tokens, client=_client, model=_model)
                            compressed_chunks.append(compressed_chunk)
                        compressed = merge_chunks(compressed_chunks)
                        output_tokens = estimate_tokens(compressed)
                    chunks_used = len(chunks)
                else:
                    raise
        else:
            chunks = chunk_text(text, max_tokens=max_chunk_tokens)
            if len(chunks) == 1:
                compressed = compress_with_llm(chunks[0], max_tokens=max_output_tokens, client=_client, model=_model)
                output_tokens = estimate_tokens(compressed)
            else:
                compressed_chunks = []
                for i, chunk in enumerate(chunks):
                    logger.info("  分段 %d/%d (%d tokens)", i + 1, len(chunks), estimate_tokens(chunk))
                    compressed_chunk = compress_with_llm(chunk, max_tokens=max_output_tokens, client=_client, model=_model)
                    compressed_chunks.append(compressed_chunk)
                compressed = merge_chunks(compressed_chunks)
                output_tokens = estimate_tokens(compressed)
            chunks_used = len(chunks)

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compressed)

        ratio = output_tokens / input_tokens if input_tokens > 0 else 0

        with lock:
            state[filename] = {
                "status": "completed",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "chunks": chunks_used,
                "compression_ratio": round(ratio, 3),
            }

        return {
            "file": filename,
            "status": "completed",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "chunks": chunks_used,
            "ratio": ratio,
        }

    except Exception as e:
        with lock:
            state[filename] = {"status": "failed", "error": str(e)}
        return {"file": filename, "status": "failed", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  对比统计函数
# ═══════════════════════════════════════════════════════════════════════════════

GARBLED_CHARS = ''.join(chr(cp) for cp in range(0x2500, 0x25A0))
GARBLED_PATTERN = re.compile(f'[{re.escape(GARBLED_CHARS)}]')
YI_SYLLABLE_PATTERN = re.compile(r'[\uA000-\uA48F]')


def analyze_text_quality(text: str) -> dict:
    """分析文本质量指标。"""
    total_chars = len(text)
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    lines = text.split('\n')
    non_empty_lines = [l for l in lines if l.strip()]
    empty_lines = [l for l in lines if not l.strip()]

    h1_count = len(re.findall(r'^#\s', text, re.MULTILINE))
    h2_count = len(re.findall(r'^##\s', text, re.MULTILINE))
    h3_count = len(re.findall(r'^###\s', text, re.MULTILINE))

    garbled_chars = len(GARBLED_PATTERN.findall(text))
    yi_syllables = len(YI_SYLLABLE_PATTERN.findall(text))

    table_rows = len(re.findall(r'^\|.*\|$', text, re.MULTILINE))

    ics_spacing = len(re.findall(r'I C S', text))
    pagebreak = len(re.findall(r'---\s*pagebreak', text, re.IGNORECASE))
    img_placeholder = len(re.findall(r'<!--.*?-->', text))

    return {
        "total_chars": total_chars,
        "chinese_chars": chinese_chars,
        "chinese_ratio": round(chinese_chars / total_chars, 4) if total_chars > 0 else 0,
        "total_lines": len(lines),
        "non_empty_lines": len(non_empty_lines),
        "empty_lines": len(empty_lines),
        "empty_line_ratio": round(len(empty_lines) / len(lines), 4) if lines else 0,
        "h1_count": h1_count,
        "h2_count": h2_count,
        "h3_count": h3_count,
        "total_headings": h1_count + h2_count + h3_count,
        "table_rows": table_rows,
        "garbled_chars": garbled_chars,
        "yi_syllables": yi_syllables,
        "ics_spacing_noise": ics_spacing,
        "pagebreak_noise": pagebreak,
        "img_placeholder_noise": img_placeholder,
        "total_noise": garbled_chars + yi_syllables + ics_spacing + pagebreak + img_placeholder,
    }


def compare_file_pair(original_path: Path, compressed_path: Path) -> dict:
    """对比单个文件的原始与压缩版本。"""
    orig_text = read_md(original_path)
    comp_text = read_md(compressed_path)

    orig_quality = analyze_text_quality(orig_text)
    comp_quality = analyze_text_quality(comp_text)
    orig_tokens = estimate_tokens(orig_text)
    comp_tokens = estimate_tokens(comp_text)
    orig_size = original_path.stat().st_size
    comp_size = compressed_path.stat().st_size

    token_ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 0
    noise_delta = comp_quality["total_noise"] - orig_quality["total_noise"]
    heading_delta = comp_quality["total_headings"] - orig_quality["total_headings"]

    return {
        "file": original_path.name,
        "original": {"size_bytes": orig_size, "tokens": orig_tokens, **orig_quality},
        "compressed": {"size_bytes": comp_size, "tokens": comp_tokens, **comp_quality},
        "diff": {
            "size_bytes": comp_size - orig_size,
            "tokens": comp_tokens - orig_tokens,
            "token_ratio": round(token_ratio, 4),
            "size_ratio": round(comp_size / orig_size, 4) if orig_size > 0 else 0,
            "noise_change": noise_delta,
            "heading_change": heading_delta,
            "chinese_ratio_change": round(comp_quality["chinese_ratio"] - orig_quality["chinese_ratio"], 4),
        },
    }


def print_compare_report(report: dict):
    """打印对比报告表格。"""
    files = report["files"]
    summary = report["summary"]

    logger.info("")
    logger.info("=" * 80)
    logger.info("原始 vs LLM 压缩 对比报告")
    logger.info("=" * 80)

    logger.info("")
    logger.info("📋 总体概览")
    logger.info("─" * 40)
    logger.info("  对比文件数:      %d", summary["total_files"])
    logger.info("  原始总大小:      %.1f KB", summary["total_orig_size"] / 1024)
    logger.info("  压缩总大小:      %.1f KB", summary["total_comp_size"] / 1024)
    logger.info("  原始总 token:    %s", f"{summary['total_orig_tokens']:,}")
    logger.info("  压缩总 token:    %s", f"{summary['total_comp_tokens']:,}")
    logger.info("  平均压缩率:      %.1f%%", summary["avg_token_ratio"] * 100)
    logger.info("  压缩节省:        %s tokens", f"{summary['total_tokens_saved']:,}")
    logger.info("  总噪声去除:      %d 处", summary["total_noise_removed"])

    q = summary["quality"]
    logger.info("")
    logger.info("📈 质量指标变化 (平均)")
    logger.info("─" * 60)
    logger.info("  %-24s %10s %10s %10s", "指标", "原始", "压缩", "变化")
    logger.info("  %s", "─" * 56)
    logger.info("  %-24s %9.1f%% %9.1f%% %+9.1f%%", "中文占比", q["avg_orig_chinese_ratio"] * 100, q["avg_comp_chinese_ratio"] * 100, q["avg_chinese_ratio_change"] * 100)
    logger.info("  %-24s %9.1f%% %9.1f%% %+9.1f%%", "空行占比", q["avg_orig_empty_line_ratio"] * 100, q["avg_comp_empty_line_ratio"] * 100, q["avg_empty_line_ratio_change"] * 100)
    logger.info("  %-24s %10.1f %10.1f %+10.1f", "各级标题数", q["avg_orig_headings"], q["avg_comp_headings"], q["avg_heading_change"])
    logger.info("  %-24s %10.1f %10.1f %+10.1f", "表格行数", q["avg_orig_table_rows"], q["avg_comp_table_rows"], q["avg_table_rows_change"])
    logger.info("  %-24s %10.1f %10.1f %+10.1f", "总噪声", q["avg_orig_noise"], q["avg_comp_noise"], q["avg_noise_change"])

    logger.info("")
    logger.info("📄 文件明细")
    logger.info("─" * 80)
    logger.info("  %-40s %10s %10s %8s %6s", "文件名", "原始token", "压缩token", "压缩率", "噪声")
    logger.info("  %s", "─" * 76)
    for f in files:
        d = f["diff"]
        noise_str = f"{d['noise_change']:+d}" if d['noise_change'] != 0 else "  0"
        logger.info("  %-38s %10s %10s %7.1f%% %6s",
                     f["file"][:38],
                     f"{f['original']['tokens']:,}",
                     f"{f['compressed']['tokens']:,}",
                     d["token_ratio"] * 100,
                     noise_str)
    logger.info("  %s", "─" * 76)
    logger.info("  %-40s %10s %10s %7.1f%% %+6d",
                 "合计",
                 f"{summary['total_orig_tokens']:,}",
                 f"{summary['total_comp_tokens']:,}",
                 summary["avg_token_ratio"] * 100,
                 summary["total_noise_removed"])


def generate_compare_report(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    report_file: Path | None = None,
) -> dict:
    """生成原始与压缩版本的完整对比报告。"""
    if input_dir is None:
        input_dir = _DEFAULT_INPUT_DIR
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    if report_file is None:
        report_file = _DEFAULT_REPORT_FILE

    if not input_dir.exists():
        logger.error("原始文件目录不存在: %s", input_dir)
        return {}
    if not output_dir.exists():
        logger.error("压缩文件目录不存在: %s", output_dir)
        return {}

    orig_files = {f.name: f for f in sorted(input_dir.glob("*.md"))}
    comp_files = {f.name: f for f in sorted(output_dir.glob("*.md"))}

    common = sorted(set(orig_files) & set(comp_files))
    missing_comp = sorted(set(orig_files) - set(comp_files))
    missing_orig = sorted(set(comp_files) - set(orig_files))

    if not common:
        logger.error("没有找到可对比的文件对（原始与压缩均需存在）")
        return {}

    logger.info("正在对比 %d 个文件...", len(common))

    file_reports = []
    total_orig_tokens = 0
    total_comp_tokens = 0
    total_orig_size = 0
    total_comp_size = 0
    total_orig_noise = 0
    total_comp_noise = 0
    total_orig_heading = 0
    total_comp_heading = 0
    total_orig_chinese_ratio = 0.0
    total_comp_chinese_ratio = 0.0
    total_orig_empty_line_ratio = 0.0
    total_comp_empty_line_ratio = 0.0
    total_orig_table_rows = 0
    total_comp_table_rows = 0

    for i, fname in enumerate(common, 1):
        result = compare_file_pair(orig_files[fname], comp_files[fname])
        file_reports.append(result)

        total_orig_tokens += result["original"]["tokens"]
        total_comp_tokens += result["compressed"]["tokens"]
        total_orig_size += result["original"]["size_bytes"]
        total_comp_size += result["compressed"]["size_bytes"]
        total_orig_noise += result["original"]["total_noise"]
        total_comp_noise += result["compressed"]["total_noise"]
        total_orig_heading += result["original"]["total_headings"]
        total_comp_heading += result["compressed"]["total_headings"]
        total_orig_chinese_ratio += result["original"]["chinese_ratio"]
        total_comp_chinese_ratio += result["compressed"]["chinese_ratio"]
        total_orig_empty_line_ratio += result["original"]["empty_line_ratio"]
        total_comp_empty_line_ratio += result["compressed"]["empty_line_ratio"]
        total_orig_table_rows += result["original"]["table_rows"]
        total_comp_table_rows += result["compressed"]["table_rows"]

        if i % 20 == 0 or i == len(common):
            logger.info("  已对比 %d/%d...", i, len(common))

    n = len(common)
    report = {
        "summary": {
            "total_files": n,
            "total_orig_tokens": total_orig_tokens,
            "total_comp_tokens": total_comp_tokens,
            "total_tokens_saved": total_orig_tokens - total_comp_tokens,
            "avg_token_ratio": round(total_comp_tokens / total_orig_tokens, 4) if total_orig_tokens > 0 else 0,
            "total_orig_size": total_orig_size,
            "total_comp_size": total_comp_size,
            "total_noise_removed": total_orig_noise - total_comp_noise,
            "missing_compressed": missing_comp,
            "missing_original": missing_orig,
            "quality": {
                "avg_orig_chinese_ratio": round(total_orig_chinese_ratio / n, 4),
                "avg_comp_chinese_ratio": round(total_comp_chinese_ratio / n, 4),
                "avg_chinese_ratio_change": round((total_comp_chinese_ratio - total_orig_chinese_ratio) / n, 4),
                "avg_orig_empty_line_ratio": round(total_orig_empty_line_ratio / n, 4),
                "avg_comp_empty_line_ratio": round(total_comp_empty_line_ratio / n, 4),
                "avg_empty_line_ratio_change": round((total_comp_empty_line_ratio - total_orig_empty_line_ratio) / n, 4),
                "avg_orig_headings": round(total_orig_heading / n, 2),
                "avg_comp_headings": round(total_comp_heading / n, 2),
                "avg_heading_change": round((total_comp_heading - total_orig_heading) / n, 2),
                "avg_orig_table_rows": round(total_orig_table_rows / n, 2),
                "avg_comp_table_rows": round(total_comp_table_rows / n, 2),
                "avg_table_rows_change": round((total_comp_table_rows - total_orig_table_rows) / n, 2),
                "avg_orig_noise": round(total_orig_noise / n, 2),
                "avg_comp_noise": round(total_comp_noise / n, 2),
                "avg_noise_change": round((total_comp_noise - total_orig_noise) / n, 2),
            },
        },
        "files": file_reports,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("  报告已保存: %s", report_file)

    return report


# ═══════════════════════════════════════════════════════════════════════════════
#  LLMCompressor 包装类
# ═══════════════════════════════════════════════════════════════════════════════

class LLMCompressor:
    """LLM Markdown 压缩工具的易用封装。

    Args:
        input_dir: 输入 Markdown 目录。
        output_dir: 输出压缩后 Markdown 目录。
        state_file: 状态文件路径（断点续传）。
        report_file: 对比报告保存路径。
        api_key: OpenAI API key（默认从环境变量读取）。
        base_url: API base URL（默认从环境变量读取）。
        model: LLM 模型名（默认从环境变量读取）。
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        state_file: str | Path | None = None,
        report_file: str | Path | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file) if state_file else self.output_dir.parent / "llm_compress_state.json"
        self.report_file = Path(report_file) if report_file else self.output_dir.parent / "llm_compress_report.json"

        _api_key = api_key or LLM_API_KEY or os.environ.get("OPENAI_API_KEY", "")
        _base_url = base_url or LLM_BASE_URL or os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        _model = model or LLM_MODEL or os.environ.get("LLM_MODEL", "deepseek-v4-flash")

        if not _api_key:
            raise ValueError("未设置 OPENAI_API_KEY，请在 .env 或环境变量中配置")

        self.client = OpenAI(base_url=_base_url, api_key=_api_key)
        self.model = _model

    def run(
        self,
        file_range: str | None = None,
        dry_run: bool = False,
        max_chunk_tokens: int = 24000,
        max_output_tokens: int = 8192,
        concurrency: int = 1,
        resume: bool = True,
        auto_compare: bool = True,
        no_chunk: bool = False,
    ) -> dict[str, Any]:
        """运行 LLM 压缩流程。

        Args:
            no_chunk: True 时不分割文本，整篇文档一次发给 LLM。

        Returns:
            处理结果汇总字典。
        """
        if not self.input_dir.exists():
            logger.error("输入目录不存在: %s", self.input_dir)
            return {"error": "input_dir not found"}

        all_files = sorted(self.input_dir.glob("*.md"))
        if not all_files:
            logger.error("输入目录中没有 .md 文件: %s", self.input_dir)
            return {"error": "no md files found"}

        md_files = self._resolve_file_range(all_files, file_range)

        logger.info("")
        logger.info("=" * 60)
        logger.info("LLM Markdown 压缩与质量提升")
        logger.info("=" * 60)
        logger.info("输入:        %s (%d 文件)", self.input_dir, len(md_files))
        logger.info("输出:        %s", self.output_dir)
        logger.info("LLM 模型:    %s", self.model)
        logger.info("并发数:      %d", concurrency)
        logger.info("分块:        %s", "否 (整篇发送)" if no_chunk else f"是 (最大 {max_chunk_tokens} tokens)")
        logger.info("=" * 60)

        if dry_run:
            self._dry_run(md_files)
            return {"status": "dry_run"}

        state = {} if not resume else self._load_state()

        resume_count = sum(
            1 for fp in md_files
            if state.get(fp.name, {}).get("status") == "completed"
            or (resume and (self.output_dir / fp.name).exists() and (self.output_dir / fp.name).stat().st_size > 100)
        )

        completed_count = 0
        failed_count = 0
        total_input_tokens = 0
        total_output_tokens = 0
        lock = Lock()

        if resume_count > 0 and resume:
            logger.info("断点续传：跳过 %d 个已完成文件", resume_count)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for fp in md_files:
                fn = fp.name
                if resume and state.get(fn, {}).get("status") == "completed":
                    continue
                # 输出文件已存在也跳过
                if resume and (self.output_dir / fn).exists() and (self.output_dir / fn).stat().st_size > 100:
                    continue
                from functools import partial
                submit_fn = partial(
                    process_file, fp, self.output_dir,
                    max_chunk_tokens, max_output_tokens,
                    state, lock, self.client, self.model,
                    no_chunk=no_chunk,
                )
                future = executor.submit(submit_fn)
                futures[future] = fn

            for i, future in enumerate(as_completed(futures), 1):
                fn = futures[future]
                result = future.result()
                status = result["status"]

                if status == "completed":
                    completed_count += 1
                    total_input_tokens += result["input_tokens"]
                    total_output_tokens += result["output_tokens"]
                    ratio = result.get("ratio", 0)
                    chunks = result.get("chunks", 1)
                    logger.info("[%d/%d] %s: in=%s out=%s (%.0f%%) ch=%d",
                                 i, len(futures), fn[:55],
                                 f"{result['input_tokens']:,}",
                                 f"{result['output_tokens']:,}",
                                 ratio * 100, chunks)
                elif status == "skipped":
                    reason = result.get("reason", "")
                    if reason != "already completed":
                        logger.info("[%d/%d] %s (%s)", i, len(futures), fn[:55], reason)
                elif status == "failed":
                    failed_count += 1
                    logger.error("[%d/%d] %s: %s", i, len(futures), fn[:55], result.get("error", ""))

                if i % 5 == 0 or status == "failed" or i == len(futures):
                    self._save_state(state)

        total_processed = completed_count + resume_count
        total_files = len(md_files)
        cost = calculate_cost(total_input_tokens, total_output_tokens)

        summary = {
            "total_files": total_files,
            "completed": completed_count,
            "resumed": resume_count,
            "failed": failed_count,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "compression_ratio": total_output_tokens / total_input_tokens if total_input_tokens > 0 else 0,
            "cost_yuan": round(cost, 4),
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("处理完成汇总")
        logger.info("=" * 60)
        logger.info("总文件数:       %d", total_files)
        logger.info("新增压缩:       %d", completed_count)
        logger.info("断点续传跳过:   %d", resume_count)
        logger.info("失败:           %d", failed_count)
        logger.info("输入 token:     %s", f"{total_input_tokens:,}")
        logger.info("输出 token:     %s", f"{total_output_tokens:,}")
        if total_input_tokens > 0:
            logger.info("压缩比:         %.1f%%", total_output_tokens / total_input_tokens * 100)
        logger.info("预估费用:       ¥%.4f", cost)
        logger.info("输出目录:       %s", self.output_dir)
        logger.info("=" * 60)

        if auto_compare and (completed_count > 0 or resume_count > 0):
            report = generate_compare_report(self.input_dir, self.output_dir, self.report_file)
            if report:
                print_compare_report(report)

        return summary

    def _resolve_file_range(
        self, all_files: list[Path], file_range: str | None
    ) -> list[Path]:
        """解析 --files 参数，如 '1-5', '3,7,9', '1-5,8,10-12'。"""
        if not file_range:
            return all_files

        selected = set()
        for part in file_range.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                a, b = int(a.strip()), int(b.strip())
                for i in range(a, b + 1):
                    if 1 <= i <= len(all_files):
                        selected.add(i - 1)
            else:
                i = int(part)
                if 1 <= i <= len(all_files):
                    selected.add(i - 1)
        return [all_files[i] for i in sorted(selected)]

    def _dry_run(self, md_files: list[Path]):
        """估算 token 用量和费用。"""
        total_input = 0
        total_output_est = 0
        file_details = []

        for fp in md_files:
            text = read_md(fp)
            tokens = estimate_tokens(text)
            total_input += tokens
            out_est = int(tokens * 0.4)
            total_output_est += out_est
            file_details.append((fp.name, tokens, out_est))

        cost = calculate_cost(total_input, total_output_est)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Dry Run 预估")
        logger.info("=" * 60)
        logger.info("文件数:           %d", len(md_files))
        logger.info("输入总 token:     %s (~%.1fK)", f"{total_input:,}", total_input / 1000)
        logger.info("预估输出 token:   %s (~%.1fK)", f"{total_output_est:,}", total_output_est / 1000)
        logger.info("预估费用:         ¥%.4f", cost)
        logger.info("模型:             %s", self.model)
        logger.info("=" * 60)

        logger.info("文件明细:")
        for name, inp, out in file_details:
            logger.info("  %-50s in=%s out≈%s", name[:50], f"{inp:,}", f"{out:,}")

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass
        return {}

    def _save_state(self, state: dict):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def compare(self) -> dict:
        """仅运行对比分析。"""
        report = generate_compare_report(self.input_dir, self.output_dir, self.report_file)
        if report:
            print_compare_report(report)
        return report or {}


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="调用远端 LLM 对 Markdown 文档进行智能压缩与质量提升"
    )
    parser.add_argument("--input-dir", default=None, help="输入目录（默认: env LLM_COMPRESS_INPUT_DIR 或 ./output/filtered_md）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认: env LLM_COMPRESS_OUTPUT_DIR 或 ./output/llm_compressed）")
    parser.add_argument("--files", type=str, default=None, help="文件范围，如 '1-5', '3,7,9', '1-5,8,10-12'")
    parser.add_argument("--dry-run", action="store_true", help="仅预估 token 用量和费用")
    parser.add_argument("--max-chunk-tokens", type=int, default=24000, help="每段最大 token 数 (默认: 24000)")
    parser.add_argument("--max-output-tokens", type=int, default=8192, help="每段最大输出 token 数 (默认: 8192)")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数 (默认: 1)")
    parser.add_argument("--resume", action="store_true", default=True, help="断点续传")
    parser.add_argument("--no-resume", action="store_true", help="关闭断点续传")
    parser.add_argument("--no-chunk", action="store_true", help="不分块，整篇文档一次发送给 LLM")
    parser.add_argument("--compare", action="store_true", help="对比已有压缩结果")
    parser.add_argument("--no-auto-compare", action="store_true", help="压缩后不自动对比")

    args = parser.parse_args()

    input_dir = args.input_dir or str(_DEFAULT_INPUT_DIR)
    output_dir = args.output_dir or str(_DEFAULT_OUTPUT_DIR)

    compressor = LLMCompressor(
        input_dir=input_dir,
        output_dir=output_dir,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL"),
    )

    if args.compare:
        compressor.compare()
        return

    compressor.run(
        file_range=args.files,
        dry_run=args.dry_run,
        max_chunk_tokens=args.max_chunk_tokens,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        no_chunk=args.no_chunk,
        auto_compare=not args.no_auto_compare,
    )


if __name__ == "__main__":
    main()
