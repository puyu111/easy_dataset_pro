"""利用 datatrove 对 Markdown 文档进行数据质量评估与过滤。

功能:
  1. 读取 Markdown 文件目录
  2. 对每个文档计算多项质量指标
  3. 根据质量指标进行过滤
  4. 输出质量报告 JSONL 和过滤后的高质量文档
  5. 生成质量统计摘要

用法:
    python -m graphrag.src.preprocessing.quality_pipeline                     # 默认模式：评估+过滤+输出
    python -m graphrag.src.preprocessing.quality_pipeline --mode stats        # 仅统计
    python -m graphrag.src.preprocessing.quality_pipeline --mode filter       # 仅过滤
    python -m graphrag.src.preprocessing.quality_pipeline --min-chars 500     # 自定义阈值
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ── 将 datatrove 加入 sys.path ──────────────────────────────────────────────
# 优先从环境变量 DATATROVE_DIR 读取，否则尝试默认路径
_DEFAULT_DATATROVE_DIR = os.environ.get(
    "DATATROVE_DIR",
    str(Path.home() / "git_project" / "datatrove"),
)


def _ensure_datatrove(datatrove_dir: str | None = None) -> None:
    """将 datatrove 目录（含 src/）加入 sys.path。"""
    d = datatrove_dir or _DEFAULT_DATATROVE_DIR
    d_path = Path(d)
    if d_path.exists() and str(d_path) not in sys.path:
        sys.path.insert(0, str(d_path))
    # datatrove 使用 src/ layout
    src_path = d_path / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


# 在模块加载时默认尝试
_ensure_datatrove()


from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from datatrove.utils.typeshelper import Languages, StatHints

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  类定义：保持与原 data_quality_pipeline.py 完全一致
# ═══════════════════════════════════════════════════════════════════════════════

class MarkdownReader(BaseDiskReader):
    """读取 .md 文件，每个文件作为一个 Document。"""

    type = "📄 - MARKDOWN READER"

    def __init__(
        self,
        data_folder: DataFolderLike,
        limit: int = -1,
        default_metadata: dict = None,
    ):
        super().__init__(
            data_folder=data_folder,
            limit=limit,
            glob_pattern="*.md",
            recursive=False,
            add_file_path=True,
            default_metadata=default_metadata or {},
        )

    def read_file(self, filepath: str) -> DocumentsPipeline:
        with self.data_folder.open(filepath, "rt", encoding="utf-8") as f:
            text = f.read()
        filename = os.path.basename(filepath)
        doc_id = filename.replace(".md", "")
        yield self.get_document_from_dict(
            {"text": text, "id": doc_id},
            source_file=filepath,
            id_in_file=0,
        )


class ChineseQualityScorer(PipelineStep):
    """为中文 Markdown 文档计算质量指标（不过滤，仅打分）。"""

    type = "📊 - QUALITY SCORER"

    GARBLED_CHARS = ''.join(chr(cp) for cp in range(0x2500, 0x25A0))
    GARBLED_PATTERNS = re.compile(f'[{re.escape(GARBLED_CHARS)}]')

    def __init__(self):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self._compute_quality(doc)
            self.update_doc_stats(doc)
            yield doc

    def _compute_quality(self, doc: Document):
        text = doc.text
        lines = text.splitlines()
        n_lines = len(lines)
        n_chars = len(text)

        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        chinese_ratio = chinese_chars / n_chars if n_chars > 0 else 0.0

        non_empty_lines = [l for l in lines if l.strip()]
        empty_line_ratio = 1.0 - (len(non_empty_lines) / n_lines) if n_lines > 0 else 0.0

        line_lengths = [len(l) for l in non_empty_lines]
        avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0.0
        short_line_ratio = sum(1 for l in line_lengths if l < 10) / len(line_lengths) if line_lengths else 0.0
        long_line_ratio = sum(1 for l in line_lengths if l > 200) / len(line_lengths) if line_lengths else 0.0

        line_set = set(non_empty_lines)
        repetition_line_ratio = 1.0 - (len(line_set) / len(non_empty_lines)) if non_empty_lines else 0.0

        garbled_matches = self.GARBLED_PATTERNS.findall(text)
        special_char_ratio = len(garbled_matches) / n_chars if n_chars > 0 else 0.0

        section_lines = [l for l in lines if l.strip().startswith("#")]
        section_count = len(section_lines)
        has_toc = "目" in text[:500] or "目录" in text[:500]

        score_chars = min(1.0, n_chars / 5000.0)
        score_chinese = min(1.0, chinese_ratio / 0.5)
        score_structure = min(1.0, section_count / 5.0)
        score_line = 1.0 - min(1.0, short_line_ratio)
        score_garbled = 1.0 - min(1.0, special_char_ratio * 500)
        score_repetition = 1.0 - min(1.0, repetition_line_ratio * 5)
        score_empty = 1.0 - min(1.0, empty_line_ratio * 2)

        quality_score = (
            0.25 * score_chars +
            0.15 * score_chinese +
            0.15 * score_structure +
            0.10 * score_line +
            0.15 * score_garbled +
            0.10 * score_repetition +
            0.10 * score_empty
        )

        if quality_score >= 0.8:
            quality_level = "优"
        elif quality_score >= 0.6:
            quality_level = "良"
        elif quality_score >= 0.4:
            quality_level = "中"
        else:
            quality_level = "差"

        doc.metadata["char_count"] = n_chars
        doc.metadata["chinese_char_count"] = chinese_chars
        doc.metadata["chinese_ratio"] = round(chinese_ratio, 4)
        doc.metadata["line_count"] = n_lines
        doc.metadata["avg_line_length"] = round(avg_line_length, 2)
        doc.metadata["short_line_ratio"] = round(short_line_ratio, 4)
        doc.metadata["long_line_ratio"] = round(long_line_ratio, 4)
        doc.metadata["empty_line_ratio"] = round(empty_line_ratio, 4)
        doc.metadata["repetition_line_ratio"] = round(repetition_line_ratio, 4)
        doc.metadata["special_char_ratio"] = round(special_char_ratio, 6)
        doc.metadata["section_count"] = section_count
        doc.metadata["has_toc"] = has_toc
        doc.metadata["quality_score"] = round(quality_score, 4)
        doc.metadata["quality_level"] = quality_level

        self.stat_update("total_docs")
        self.stat_update(f"quality_{quality_level}")


class ChineseQualityFilter(BaseFilter):
    """根据预计算的质量指标过滤文档。"""

    type = "🔻 - CHINESE QUALITY FILTER"

    def __init__(
        self,
        min_chars: int = 200,
        min_chinese_ratio: float = 0.01,
        max_special_char_ratio: float = 0.001,
        min_quality_score: float = 0.4,
        max_repetition_ratio: float = 0.5,
        exclusion_writer=None,
    ):
        super().__init__(exclusion_writer)
        self.min_chars = min_chars
        self.min_chinese_ratio = min_chinese_ratio
        self.max_special_char_ratio = max_special_char_ratio
        self.min_quality_score = min_quality_score
        self.max_repetition_ratio = max_repetition_ratio

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        meta = doc.metadata
        reasons = []

        if meta.get("char_count", 0) < self.min_chars:
            reasons.append(f"too_short({meta.get('char_count',0)}<{self.min_chars})")
        if meta.get("chinese_ratio", 1.0) < self.min_chinese_ratio:
            reasons.append(f"low_chinese_ratio({meta.get('chinese_ratio',0):.2f}<{self.min_chinese_ratio})")
        if meta.get("special_char_ratio", 0) > self.max_special_char_ratio:
            reasons.append(f"high_garbled({meta.get('special_char_ratio',0):.4f}>{self.max_special_char_ratio})")
        if meta.get("quality_score", 0) < self.min_quality_score:
            reasons.append(f"low_quality({meta.get('quality_score',0):.2f}<{self.min_quality_score})")
        if meta.get("repetition_line_ratio", 0) > self.max_repetition_ratio:
            reasons.append(f"high_repetition({meta.get('repetition_line_ratio',0):.2f}>{self.max_repetition_ratio})")

        if reasons:
            return False, ";".join(reasons)
        return True


class QualityStatsCollector(PipelineStep):
    """收集质量统计信息并输出摘要。"""

    type = "📈 - STATS COLLECTOR"

    def __init__(self):
        super().__init__()

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        scores = []
        levels = {"优": 0, "良": 0, "中": 0, "差": 0}

        for doc in data:
            meta = doc.metadata
            scores.append(meta.get("quality_score", 0))
            levels[meta.get("quality_level", "未知")] = levels.get(meta.get("quality_level", "未知"), 0) + 1
            self.update_doc_stats(doc)
            yield doc

        total = len(scores)
        if total > 0:
            avg = sum(scores) / total
            logger.info("\n" + "=" * 60)
            logger.info("📊 质量统计摘要")
            logger.info("=" * 60)
            logger.info("文档总数:        %d", total)
            logger.info("平均质量分:      %.4f", avg)
            logger.info("最高质量分:      %.4f", max(scores))
            logger.info("最低质量分:      %.4f", min(scores))
            logger.info("")
            logger.info("质量分布:")
            for level in ["优", "良", "中", "差"]:
                count = levels.get(level, 0)
                bar = "█" * int(count / max(1, total) * 50)
                logger.info("  %s: %3d (%5.1f%%) %s", level, count, count / max(1, total) * 100, bar)
            logger.info("=" * 60)


class MarkdownWriter(PipelineStep):
    """将 Document 文本写回 .md 文件。"""

    type = "💾 - MARKDOWN WRITER"

    def __init__(self, output_dir: Path):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            filename = f"{doc.id}.md"
            filepath = self.output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(doc.text)
            self.stat_update("files_written")
            self.update_doc_stats(doc)
            yield doc


# ═══════════════════════════════════════════════════════════════════════════════
#  QualityPipeline 包装类 —— 方便从其他模块调用
# ═══════════════════════════════════════════════════════════════════════════════

class QualityPipeline:
    """datatrove 质量管线的易用封装。

    Args:
        input_dir: Markdown 文件输入目录。
        output_base: 输出根目录（默认 input_dir 上级目录的 output/）。
        datatrove_dir: datatrove 源码目录路径。
    """

    def __init__(
        self,
        input_dir: str,
        output_base: str | None = None,
        datatrove_dir: str | None = None,
        log_file: str | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base) if output_base else self.input_dir.parent / "output"
        self.log_file = Path(log_file) if log_file else self.output_base / "quality_filter_log.txt"
        _ensure_datatrove(datatrove_dir)

        self.quality_report_dir = self.output_base / "quality_report"
        self.filtered_dir = self.output_base / "filtered_md"
        self.excluded_dir = self.output_base / "excluded"
        self.logging_dir = self.output_base / "logs"

    def run(
        self,
        mode: str = "full",
        min_chars: int = 200,
        min_quality: float = 0.4,
        tasks: int = 1,
        workers: int = 1,
    ) -> int:
        """运行质量评估流水线。

        Returns:
            过滤后的文档数（filter/full 模式），或总文档数（stats 模式）。
        """
        self.output_base.mkdir(parents=True, exist_ok=True)
        self.quality_report_dir.mkdir(parents=True, exist_ok=True)

        pipeline = self._build_pipeline(mode, min_chars, min_quality)

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=str(self.logging_dir / f"logs_{mode}"),
            tasks=tasks,
            workers=workers,
            skip_completed=False,
        )
        executor.run()

        self._print_summary()
        self._write_filter_log(mode, min_chars, min_quality)
        return self._count_filtered()

    def _build_pipeline(self, mode: str, min_chars: int, min_quality: float):
        reader = MarkdownReader(data_folder=str(self.input_dir))
        scorer = ChineseQualityScorer()
        stats_collector = QualityStatsCollector()
        report_writer = JsonlWriter(
            output_folder=str(self.quality_report_dir),
            output_filename="quality_report.jsonl",
            compression=None,
        )

        if mode == "stats":
            return [reader, scorer, stats_collector, report_writer]

        quality_filter = ChineseQualityFilter(
            min_chars=min_chars,
            min_quality_score=min_quality,
        )
        exclusion_writer = JsonlWriter(
            output_folder=str(self.excluded_dir),
            output_filename="excluded.jsonl",
            compression=None,
        )
        quality_filter.exclusion_writer = exclusion_writer
        filtered_writer = MarkdownWriter(self.filtered_dir)

        return [reader, scorer, stats_collector, report_writer, quality_filter, filtered_writer]

    def _print_summary(self):
        """打印最终结果汇总。"""
        report_file = self.quality_report_dir / "quality_report.jsonl"
        if not report_file.exists():
            logger.warning("质量报告未生成")
            return

        docs = []
        with open(report_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

        if not docs:
            logger.warning("质量报告为空")
            return

        total = len(docs)
        levels = {}
        for d in docs:
            meta = d.get("metadata", {})
            lv = meta.get("quality_level", "未知")
            levels[lv] = levels.get(lv, 0) + 1

        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 处理完成汇总")
        logger.info("=" * 60)
        logger.info("文档总数:        %d", total)
        logger.info("报告文件:        %s", report_file)
        filtered_count = self._count_filtered()
        logger.info("高质量文档:      %d (保存至 %s)", filtered_count, self.filtered_dir)
        logger.info("")
        logger.info("质量分布:")
        for level in ["优", "良", "中", "差"]:
            count = levels.get(level, 0)
            pct = count / total * 100 if total > 0 else 0
            logger.info("  %s: %3d (%5.1f%%)", level, count, pct)
        logger.info("=" * 60)

    def _write_filter_log(self, mode: str, min_chars: int, min_quality: float):
        """将过滤详情写入日志文件。"""
        if mode == "stats":
            return

        report_file = self.quality_report_dir / "quality_report.jsonl"
        excluded_file = self.excluded_dir / "excluded.jsonl"

        all_scores: dict[str, dict] = {}
        if report_file.exists():
            with open(report_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        all_scores[doc["id"]] = doc.get("metadata", {})

        excluded_reasons: dict[str, str] = {}
        if excluded_file.exists():
            with open(excluded_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        reasons = doc.get("metadata", {}).get("filter_reason", "")
                        excluded_reasons[doc["id"]] = reasons

        if not all_scores:
            return

        passed = [f for f in all_scores if f not in excluded_reasons]
        filtered = list(excluded_reasons.keys())

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as log:
            log.write(f"{'='*70}\n")
            log.write(f"Markdown 质量过滤详情\n")
            log.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"输入目录: {self.input_dir}\n")
            log.write(f"输出目录: {self.filtered_dir}\n")
            log.write(f"阈值: 最少字符={min_chars}, 最低质量分={min_quality}\n")
            log.write(f"{'='*70}\n\n")

            log.write(f"=== 通过的文件 ({len(passed)} 个) ===\n")
            log.write(f"{'文件名':<50} {'字符数':>8} {'质量分':>7} {'等级'}\n")
            log.write(f"{'─'*70}\n")
            for fname in sorted(passed, key=lambda x: all_scores[x].get("quality_score", 0), reverse=True):
                m = all_scores[fname]
                log.write(f"{fname[:48]:48s} {m.get('char_count', 0):>8d} {m.get('quality_score', 0):>7.3f}  {m.get('quality_level', '?')}\n")

            if filtered:
                log.write(f"\n=== 过滤掉的文件 ({len(filtered)} 个) ===\n")
                log.write(f"{'文件名':<50} {'字符数':>8} {'质量分':>7} {'等级':>4}  原因\n")
                log.write(f"{'─'*70}\n")
                for fname in sorted(filtered):
                    m = all_scores.get(fname, {})
                    reason = excluded_reasons.get(fname, "")
                    log.write(f"{fname[:48]:48s} {m.get('char_count', 0):>8d} {m.get('quality_score', 0):>7.3f}  {m.get('quality_level', '?'):>4}  {reason[:60]}\n")

            log.write(f"\n{'='*70}\n")
            log.write(f"汇总: {len(passed)} 通过, {len(filtered)} 过滤 (共 {len(all_scores)} 个)\n")
            log.write(f"输出目录: {self.filtered_dir}\n")
            log.write(f"日志文件: {self.log_file}\n")
            log.write(f"{'='*70}\n")

        logger.info("  过滤详情日志: %s", self.log_file)

    def _count_filtered(self) -> int:
        return len(list(self.filtered_dir.glob("*.md"))) if self.filtered_dir.exists() else 0


def build_pipeline(
    mode: str,
    min_chars: int = 200,
    min_quality: float = 0.4,
    input_dir: str | None = None,
    output_base: str | None = None,
) -> list:
    """构建 datatrove pipeline（向后兼容）。"""
    qp = QualityPipeline(
        input_dir=input_dir or ".",
        output_base=output_base,
    )
    return qp._build_pipeline(mode, min_chars, min_quality)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="利用 datatrove 对 Markdown 文档进行数据质量评估与过滤"
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Markdown 输入目录（默认: 当前目录下的 documents_md）",
    )
    parser.add_argument(
        "--output-base", default=None,
        help="输出根目录（默认: input_dir 上级目录的 output）",
    )
    parser.add_argument(
        "--datatrove-dir", default=None,
        help="datatrove 源码目录路径",
    )
    parser.add_argument(
        "--mode", choices=["stats", "filter", "full"], default="full",
        help="运行模式: stats=仅统计, filter=仅过滤, full=完整流程 (默认)",
    )
    parser.add_argument(
        "--min-chars", type=int, default=200, help="最小字符数阈值 (默认: 200)",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.4, help="最低质量分数阈值 0~1 (默认: 0.4)",
    )
    parser.add_argument("--tasks", type=int, default=1, help="任务分片数 (默认: 1)")
    parser.add_argument("--workers", type=int, default=1, help="并行工作进程数 (默认: 1)")

    args = parser.parse_args()

    input_dir = args.input_dir or str(Path.cwd() / "documents_md")
    output_base = args.output_base or str(Path(input_dir).parent / "output")

    qp = QualityPipeline(
        input_dir=input_dir,
        output_base=output_base,
        datatrove_dir=args.datatrove_dir,
    )
    qp.run(
        mode=args.mode,
        min_chars=args.min_chars,
        min_quality=args.min_quality,
        tasks=args.tasks,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
