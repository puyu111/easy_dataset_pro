"""基于 LLM 的数据集生成模块：从压缩后的 Markdown 文档中生成问答对。

功能:
  1. 读取指定目录下的 Markdown 文件
  2. 将文件内容发送给远端 LLM，生成高质量问答对
  3. 对超长文档自动分段处理
  4. 结果保存为 JSONL 格式
  5. 支持断点续传、错误重试

用法:
    python -m graphrag.src.preprocessing.dataset_generator                           # 全部生成
    python -m graphrag.src.preprocessing.dataset_generator --files 1-5               # 指定范围
    python -m graphrag.src.preprocessing.dataset_generator --dry-run                 # 仅预估
    python -m graphrag.src.preprocessing.dataset_generator --questions-per-chunk 5   # 每段生成 5 个问题

配置:
  从 .env 文件或环境变量读取:
    OPENAI_API_KEY=sk-xxx
    LLM_BASE_URL=https://api.deepseek.com
    LLM_MODEL=deepseek-v4-flash
"""

import argparse
import json
import logging
import os
import re
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
_DEFAULT_INPUT_DIR = Path(os.environ.get(
    "DATASET_INPUT_DIR",
    str(Path.cwd() / "output" / "llm_compressed"),
))
_DEFAULT_OUTPUT_DIR = Path(os.environ.get(
    "DATASET_OUTPUT_DIR",
    str(Path.cwd() / "output" / "dataset"),
))
_DEFAULT_STATE_FILE = Path(os.environ.get(
    "DATASET_STATE_FILE",
    str(Path.cwd() / "output" / "dataset_state.json"),
))

# ── LLM 配置 ─────────────────────────────────────────────────────────────────
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

# ── 生成提示词 ───────────────────────────────────────────────────────────────
GENERATE_SYSTEM_PROMPT = """你是一名专业的数据集构建专家，擅长从技术文档中生成高质量的检索问题。

## 任务

根据用户提供的技术文档内容，生成尽可能多的高质量问题，用于构建微调数据集。

## 要求

1. **问题质量**：
   - 问题应基于文档的实际内容，答案可从文中找到
   - 问题类型多样化：事实型、定义型、比较型、推理型、列举型
   - 问题应自然、具体，避免过于宽泛或模糊
   - 问题应覆盖文档的不同段落和主题，尽可能全面

2. **问题必须自包含**：
   - 问题中必须包含文件名中的关键标识信息，使问题脱离上下文也能明确知道在问哪个文档
   - 例如：文件名为"10kV配电变压器能效GBT32893.md"，不要问"附录C列举了哪几种管理模式？"，应问"10kV配电变压器能效GBT32893 的附录C列举了哪几种变电站运行管理模式？"

3. **数量要求**：
   - 根据文档内容的丰富程度，尽可能多地生成问题
   - 不要遗漏重要信息点，每个章节、每个关键参数、每个定义都应有对应问题

4. **格式要求**：
   严格按照以下 JSON 数组格式输出，不要包含任何其他内容:
   ```json
   ["问题1", "问题2", "问题3"]
   ```

5. **注意事项**：
   - 不要生成与文档内容无关的问题
   - 不要重复生成相似的问题
   - 直接输出 JSON 字符串数组，不要包含解释或标记
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。"""
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    ascii_chars = len(text) - chinese_chars
    return int(chinese_chars * 1.5 + ascii_chars * 0.25)


def read_md(filepath: Path) -> str:
    """读取 Markdown 文件。"""
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


def parse_qa_response(response: str) -> list[str]:
    """从 LLM 响应中解析问题列表（字符串数组）。"""
    text = response.strip()
    # 去除 markdown 代码块标记
    if text.startswith("```"):
        text = re.sub(r'^```\w*\n?', '', text)
        text = re.sub(r'\n?```$', '', text)
        text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [item.strip() for item in result if isinstance(item, str) and item.strip()]
    except json.JSONDecodeError:
        pass

    # 尝试提取 JSON 数组部分
    match = re.search(r'\[[\s\S]*\]', text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [item.strip() for item in result if isinstance(item, str) and item.strip()]
        except json.JSONDecodeError:
            pass

    logger.warning("无法解析响应: %s...", text[:200])
    return []


def generate_qa_with_llm(
    text: str,
    questions_per_chunk: int = 5,
    retries: int = 3,
    client: OpenAI | None = None,
    model: str | None = None,
    filename: str = "",
) -> list[str]:
    """发送文本到 LLM 生成检索问题。"""
    if not text.strip():
        return []

    _client = client or OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    _model = model or LLM_MODEL

    doc_hint = f"文档文件名: {filename}\n\n" if filename else ""
    user_prompt = f"{doc_hint}请根据以下文档内容生成尽可能多的高质量检索问题（问题中必须包含文件名标识）：\n\n{text}"

    for attempt in range(retries):
        try:
            resp = _client.chat.completions.create(
                model=_model,
                messages=[
                    {"role": "system", "content": GENERATE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            content = resp.choices[0].message.content.strip()
            questions = parse_qa_response(content)
            return questions
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

    return []


def process_file(
    filepath: Path,
    output_dir: Path,
    questions_per_chunk: int,
    max_chunk_tokens: int,
    state: dict,
    lock: Lock,
    client: OpenAI | None = None,
    model: str | None = None,
    no_chunk: bool = False,
) -> dict:
    """处理单个文件：读取 -> (可选分块) -> LLM 生成 QA -> 保存。"""
    filename = filepath.name
    stem = filepath.stem
    output_path = output_dir / f"{stem}.jsonl"

    # 断点续传：已完成跳过
    with lock:
        if state.get(filename, {}).get("status") == "completed":
            return {"file": filename, "status": "skipped", "reason": "already completed"}

    # 输出文件已存在跳过
    if output_path.exists() and output_path.stat().st_size > 10:
        with lock:
            if filename not in state or state.get(filename, {}).get("status") != "completed":
                state[filename] = {"status": "completed", "questions": 0, "chunks": 0}
        return {"file": filename, "status": "skipped", "reason": "output already exists"}

    try:
        text = read_md(filepath)
        input_tokens = estimate_tokens(text)

        if len(text.strip()) < 100:
            with lock:
                state[filename] = {"status": "skipped", "reason": "too short"}
            return {"file": filename, "status": "skipped", "reason": "too short"}

        _client = client or OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        _model = model or LLM_MODEL

        all_qa_pairs = []

        if no_chunk:
            # 不分块：整篇一次发送；超出上下文自动回退分块
            try:
                qa_pairs = generate_qa_with_llm(
                    text, questions_per_chunk=questions_per_chunk,
                    client=_client, model=_model, filename=filename,
                )
                all_qa_pairs.extend(qa_pairs)
                chunks_used = 1
            except Exception as e:
                err_msg = str(e)
                if "maximum context length" in err_msg or "reduce the length" in err_msg:
                    logger.warning("  %s: 超出上下文限制，自动回退分块模式", filename)
                    chunks = chunk_text(text, max_tokens=max_chunk_tokens)
                    for i, chunk in enumerate(chunks):
                        logger.info("  分段 %d/%d (%d tokens)", i + 1, len(chunks), estimate_tokens(chunk))
                        qa_pairs = generate_qa_with_llm(
                            chunk, questions_per_chunk=questions_per_chunk,
                            client=_client, model=_model, filename=filename,
                        )
                        all_qa_pairs.extend(qa_pairs)
                    chunks_used = len(chunks)
                else:
                    raise
        else:
            chunks = chunk_text(text, max_tokens=max_chunk_tokens)
            chunks_used = len(chunks)
            for i, chunk in enumerate(chunks):
                if chunks_used > 1:
                    logger.info("  分段 %d/%d (%d tokens)", i + 1, chunks_used, estimate_tokens(chunk))
                qa_pairs = generate_qa_with_llm(
                    chunk, questions_per_chunk=questions_per_chunk,
                    client=_client, model=_model, filename=filename,
                )
                all_qa_pairs.extend(qa_pairs)

        # 去重
        unique_questions = list(dict.fromkeys(q.strip() for q in all_qa_pairs if q.strip()))

        # 保存为 JSONL，每行一个 JSON 对象
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for q in unique_questions:
                record = {"question": q, "source": filename}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with lock:
            state[filename] = {
                "status": "completed",
                "questions": len(unique_questions),
                "input_tokens": input_tokens,
                "chunks": chunks_used,
            }

        return {
            "file": filename,
            "status": "completed",
            "questions": len(unique_questions),
            "input_tokens": input_tokens,
            "chunks": chunks_used,
        }

    except Exception as e:
        with lock:
            state[filename] = {"status": "failed", "error": str(e)}
        return {"file": filename, "status": "failed", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  DatasetGenerator 封装类
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetGenerator:
    """基于 LLM 的数据集生成工具。

    Args:
        input_dir: 输入 Markdown 目录（默认: output/llm_compressed）。
        output_dir: 输出 JSONL 目录（默认: output/dataset）。
        state_file: 状态文件路径（断点续传）。
        questions_per_chunk: 每个文本块生成的问题数。
        max_chunk_tokens: 分块最大 token 数。
        api_key: OpenAI API key。
        base_url: API base URL。
        model: LLM 模型名。
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        state_file: str | Path | None = None,
        questions_per_chunk: int = 5,
        max_chunk_tokens: int = 24000,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.state_file = Path(state_file) if state_file else self.output_dir.parent / "dataset_state.json"
        self.questions_per_chunk = questions_per_chunk
        self.max_chunk_tokens = max_chunk_tokens

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
        concurrency: int = 1,
        resume: bool = True,
        no_chunk: bool = False,
    ) -> dict[str, Any]:
        """运行数据集生成流程。"""
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
        logger.info("LLM 数据集生成")
        logger.info("=" * 60)
        logger.info("输入:            %s (%d 文件)", self.input_dir, len(md_files))
        logger.info("输出:            %s", self.output_dir)
        logger.info("LLM 模型:        %s", self.model)
        logger.info("每段问题数:      %d", self.questions_per_chunk)
        logger.info("并发数:          %d", concurrency)
        logger.info("分块:            %s", "否 (整篇发送)" if no_chunk else f"是 (最大 {self.max_chunk_tokens} tokens)")
        logger.info("=" * 60)

        if dry_run:
            self._dry_run(md_files)
            return {"status": "dry_run"}

        state = {} if not resume else self._load_state()

        resume_count = sum(
            1 for fp in md_files
            if state.get(fp.name, {}).get("status") == "completed"
            or (resume and (self.output_dir / f"{fp.stem}.jsonl").exists()
                and (self.output_dir / f"{fp.stem}.jsonl").stat().st_size > 10)
        )

        completed_count = 0
        failed_count = 0
        total_questions = 0
        total_input_tokens = 0
        lock = Lock()

        if resume_count > 0 and resume:
            logger.info("断点续传：跳过 %d 个已完成文件", resume_count)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {}
            for fp in md_files:
                fn = fp.name
                if resume and state.get(fn, {}).get("status") == "completed":
                    continue
                if resume and (self.output_dir / f"{fp.stem}.jsonl").exists() \
                        and (self.output_dir / f"{fp.stem}.jsonl").stat().st_size > 10:
                    continue
                from functools import partial
                submit_fn = partial(
                    process_file, fp, self.output_dir,
                    self.questions_per_chunk, self.max_chunk_tokens,
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
                    total_questions += result.get("questions", 0)
                    total_input_tokens += result.get("input_tokens", 0)
                    logger.info("[%d/%d] %s: %d 问题 (ch=%d)",
                                i, len(futures), fn[:55],
                                result.get("questions", 0),
                                result.get("chunks", 1))
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

        summary = {
            "total_files": len(md_files),
            "completed": completed_count,
            "resumed": resume_count,
            "failed": failed_count,
            "total_questions": total_questions,
            "input_tokens": total_input_tokens,
        }

        logger.info("")
        logger.info("=" * 60)
        logger.info("数据集生成完成")
        logger.info("=" * 60)
        logger.info("总文件数:       %d", len(md_files))
        logger.info("新增生成:       %d", completed_count)
        logger.info("断点续传跳过:   %d", resume_count)
        logger.info("失败:           %d", failed_count)
        logger.info("总问题数:       %d", total_questions)
        logger.info("输入 token:     %s", f"{total_input_tokens:,}")
        logger.info("输出目录:       %s", self.output_dir)
        logger.info("=" * 60)

        # 生成汇总文件
        self._merge_results()

        return summary

    def _resolve_file_range(self, all_files: list[Path], file_range: str | None) -> list[Path]:
        """解析 --files 参数。"""
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
        """预估 token 用量和问题数。"""
        total_input = 0
        total_questions_est = 0

        for fp in md_files:
            text = read_md(fp)
            tokens = estimate_tokens(text)
            total_input += tokens
            chunks = chunk_text(text, max_tokens=self.max_chunk_tokens)
            total_questions_est += len(chunks) * self.questions_per_chunk

        logger.info("")
        logger.info("=" * 60)
        logger.info("Dry Run 预估")
        logger.info("=" * 60)
        logger.info("文件数:           %d", len(md_files))
        logger.info("输入总 token:     %s (~%.1fK)", f"{total_input:,}", total_input / 1000)
        logger.info("预估问题数:       ~%d", total_questions_est)
        logger.info("模型:             %s", self.model)
        logger.info("=" * 60)

    def _merge_results(self):
        """将所有 JSONL 文件合并为一个汇总文件。"""
        all_qa = []
        for jsonl_file in sorted(self.output_dir.glob("*.jsonl")):
            if jsonl_file.name == "dataset_all.jsonl":
                continue
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            all_qa.append(json.loads(line))
            except Exception as e:
                logger.warning("读取 %s 失败: %s", jsonl_file.name, e)

        if all_qa:
            merged_path = self.output_dir / "dataset_all.jsonl"
            with open(merged_path, "w", encoding="utf-8") as f:
                for qa in all_qa:
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
            logger.info("汇总文件: %s (%d 个问答对)", merged_path, len(all_qa))

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


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="基于 LLM 从 Markdown 文档生成数据集问答对"
    )
    parser.add_argument("--input-dir", default=None,
                        help="输入目录（默认: output/llm_compressed）")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（默认: output/dataset）")
    parser.add_argument("--files", type=str, default=None,
                        help="文件范围，如 '1-5', '3,7,9'")
    parser.add_argument("--questions-per-chunk", type=int, default=5,
                        help="每段生成的问题数 (默认: 5)")
    parser.add_argument("--max-chunk-tokens", type=int, default=24000,
                        help="每段最大 token 数 (默认: 24000)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="并发数 (默认: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅预估不调用 LLM")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="断点续传 (默认开启)")
    parser.add_argument("--no-resume", action="store_true",
                        help="关闭断点续传")
    parser.add_argument("--no-chunk", action="store_true",
                        help="不分块，整篇文档一次发送")

    args = parser.parse_args()

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    input_dir = args.input_dir or str(_DEFAULT_INPUT_DIR)
    output_dir = args.output_dir or str(_DEFAULT_OUTPUT_DIR)

    generator = DatasetGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        questions_per_chunk=args.questions_per_chunk,
        max_chunk_tokens=args.max_chunk_tokens,
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
        model=os.environ.get("LLM_MODEL"),
    )

    generator.run(
        file_range=args.files,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        no_chunk=args.no_chunk,
    )


if __name__ == "__main__":
    main()
