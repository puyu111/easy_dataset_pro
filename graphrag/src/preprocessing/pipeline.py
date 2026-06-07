"""预处理流水线编排器：串联文档转换 → 质量过滤 → LLM 压缩。"""

import logging
from pathlib import Path
from typing import Any

from graphrag.src.preprocessing.convert_to_md import batch_convert
from graphrag.src.preprocessing.quality_pipeline import QualityPipeline
from graphrag.src.preprocessing.llm_compress import LLMCompressor

_LOG = logging.getLogger(__name__)


class PreprocessingPipeline:
    """文档预处理的完整流水线。

    三个阶段:
      1. convert:  用 Docling 将原始文档 (PDF/DOCX/PPTX/HTML) 转为 Markdown
      2. quality:  用 datatrove 评估/过滤 Markdown 质量
      3. compress: 用 LLM 对高质量 Markdown 进行压缩清洗

    Args:
        docs_dir: 原始文档目录（输入给 convert 阶段）。
        md_dir: 第一阶段输出的 Markdown 目录。
        filtered_dir: 第二阶段过滤后的 Markdown 目录。
        compressed_dir: 第三阶段压缩后的 Markdown 目录。
        output_base: 输出根目录（如未指定具体路径则用此目录下的子目录）。
        datatrove_dir: datatrove 源码目录路径。
    """

    def __init__(
        self,
        docs_dir: str | Path = "documents",
        md_dir: str | Path | None = None,
        filtered_dir: str | Path | None = None,
        compressed_dir: str | Path | None = None,
        output_base: str | Path | None = None,
        datatrove_dir: str | Path | None = None,
    ):
        self.docs_dir = Path(docs_dir)

        if output_base:
            self.md_dir = Path(md_dir) if md_dir else Path(output_base) / "documents_md"
            self.filtered_dir = Path(filtered_dir) if filtered_dir else Path(output_base) / "filtered_md"
            self.compressed_dir = Path(compressed_dir) if compressed_dir else Path(output_base) / "llm_compressed"
        else:
            self.md_dir = Path(md_dir) if md_dir else Path(f"{docs_dir}_md")
            self.filtered_dir = Path(filtered_dir) if filtered_dir else self.md_dir.parent / "output" / "filtered_md"
            self.compressed_dir = Path(compressed_dir) if compressed_dir else self.md_dir.parent / "output" / "llm_compressed"

        self.datatrove_dir = Path(datatrove_dir) if datatrove_dir else None

    # ── 阶段 1: Docling 转换 ─────────────────────────────────────────────

    def convert(
        self,
        formats: list[str] | None = None,
        per_page: bool = False,
        image_mode: str = "embedded",
        page_breaks: bool = True,
        do_ocr: bool = True,
        force_ocr: bool = False,
        max_pages: int = 50,
        resume: bool = True,
        device: str = "auto",
        validate: bool = True,
        reconvert_bad: bool = False,
    ) -> int:
        """阶段 1: 用 Docling 将原始文档转为 Markdown。

        Returns:
            成功转换的文件数。
        """
        _LOG.info("=" * 60)
        _LOG.info("阶段 1/3: Docling 文档 → Markdown")
        _LOG.info("  输入: %s", self.docs_dir)
        _LOG.info("  输出: %s", self.md_dir)
        _LOG.info("=" * 60)

        count = batch_convert(
            input_dir=str(self.docs_dir),
            output_dir=str(self.md_dir),
            formats=formats,
            per_page=per_page,
            image_mode=image_mode,
            page_breaks=page_breaks,
            do_ocr=do_ocr,
            force_ocr=force_ocr,
            max_pages=max_pages,
            resume=resume,
            device=device,
            validate=validate,
            reconvert_bad=reconvert_bad,
        )
        _LOG.info("阶段 1 完成: %d 个文件", count)
        return count

    # ── 阶段 2: 质量过滤 ─────────────────────────────────────────────

    def quality_filter(
        self,
        mode: str = "full",
        min_chars: int = 200,
        min_quality: float = 0.4,
        tasks: int = 1,
        workers: int = 1,
    ) -> int:
        """阶段 2: 用 datatrove 评估/过滤 Markdown 质量。

        Args:
            mode: "stats"=仅统计, "filter"=仅过滤, "full"=完整流程。
            min_chars: 最小字符数阈值。
            min_quality: 最低质量分数阈值。

        Returns:
            过滤后保留的文件数。
        """
        _LOG.info("=" * 60)
        _LOG.info("阶段 2/3: Markdown 质量评估与过滤")
        _LOG.info("  输入: %s", self.md_dir)
        _LOG.info("  输出: %s", self.filtered_dir)
        _LOG.info("  模式: %s", mode)
        _LOG.info("=" * 60)

        if not self.md_dir.exists() or not list(self.md_dir.glob("*.md")):
            _LOG.warning("Markdown 目录为空或不存在，跳过质量过滤阶段")
            return 0

        qp = QualityPipeline(
            input_dir=str(self.md_dir),
            output_base=str(self.filtered_dir.parent),
            datatrove_dir=str(self.datatrove_dir) if self.datatrove_dir else None,
        )

        # 将 QualityPipeline 的输出指向 filtered_dir
        qp.filtered_dir = self.filtered_dir

        count = qp.run(
            mode=mode,
            min_chars=min_chars,
            min_quality=min_quality,
            tasks=tasks,
            workers=workers,
        )
        _LOG.info("阶段 2 完成: %d 个文件通过过滤", count)
        return count

    # ── 阶段 3: LLM 压缩 ─────────────────────────────────────────────

    def compress(
        self,
        max_chunk_tokens: int = 24000,
        max_output_tokens: int = 8192,
        concurrency: int = 1,
        resume: bool = True,
        dry_run: bool = False,
        auto_compare: bool = True,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        no_chunk: bool = False,
    ) -> dict[str, Any]:
        """阶段 3: 用 LLM 对高质量 Markdown 进行压缩清洗。

        Args:
            no_chunk: True 时不分割文本，整篇文档一次发给 LLM。
        """
        _LOG.info("=" * 60)
        _LOG.info("阶段 3/3: LLM 压缩与质量提升")
        _LOG.info("  输入: %s", self.filtered_dir)
        _LOG.info("  输出: %s", self.compressed_dir)
        _LOG.info("  分块: %s", "否 (整篇发送)" if no_chunk else f"是 (最大 {max_chunk_tokens} tokens)")
        _LOG.info("=" * 60)

        if not self.filtered_dir.exists() or not list(self.filtered_dir.glob("*.md")):
            _LOG.warning("过滤后 Markdown 目录为空或不存在，跳过 LLM 压缩阶段")
            return {"error": "no filtered md files"}

        compressor = LLMCompressor(
            input_dir=str(self.filtered_dir),
            output_dir=str(self.compressed_dir),
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

        return compressor.run(
            dry_run=dry_run,
            max_chunk_tokens=max_chunk_tokens,
            max_output_tokens=max_output_tokens,
            concurrency=concurrency,
            resume=resume,
            auto_compare=auto_compare,
            no_chunk=no_chunk,
        )

    # ── 全流程 ─────────────────────────────────────────────────────────

    def run_all(
        self,
        convert_kwargs: dict | None = None,
        quality_kwargs: dict | None = None,
        compress_kwargs: dict | None = None,
    ) -> dict[str, Any]:
        """依次执行三个阶段。

        Args:
            convert_kwargs: 传递给 convert() 的参数。
            quality_kwargs: 传递给 quality_filter() 的参数。
            compress_kwargs: 传递给 compress() 的参数。

        Returns:
            包含各阶段结果摘要的字典。
        """
        results: dict[str, Any] = {}

        # 阶段 1
        try:
            results["convert"] = self.convert(**(convert_kwargs or {}))
        except Exception as e:
            _LOG.error("阶段 1 失败: %s", e)
            results["convert"] = {"error": str(e)}

        # 阶段 2
        try:
            results["quality"] = self.quality_filter(**(quality_kwargs or {}))
        except Exception as e:
            _LOG.error("阶段 2 失败: %s", e)
            results["quality"] = {"error": str(e)}

        # 阶段 3
        try:
            results["compress"] = self.compress(**(compress_kwargs or {}))
        except Exception as e:
            _LOG.error("阶段 3 失败: %s", e)
            results["compress"] = {"error": str(e)}

        _LOG.info("=" * 60)
        _LOG.info("预处理流水线执行完成")
        _LOG.info("  Stage 1 (convert):     %s", results.get("convert"))
        _LOG.info("  Stage 2 (quality):     %s", results.get("quality"))
        _LOG.info("  Stage 3 (compress):    %s", results.get("compress", {}).get("total_files", "N/A"))
        _LOG.info("=" * 60)

        return results


def run_preprocessing(config: dict | None = None) -> dict[str, Any]:
    """便捷函数：从配置字典运行完整预处理流程。

    Args:
        config: 包含 pipeline 配置的字典。支持以下键:
            - preprocessing: 预处理配置（docs_dir, md_dir, output_base 等）
            - llm: LLM 配置（api_key, base_url, model）
            - convert / quality / compress: 各阶段额外参数
    """
    cfg = config or {}
    # 支持从顶层或 preprocessing 子键读取配置
    pre = cfg.get("preprocessing", cfg)
    pipeline = PreprocessingPipeline(
        docs_dir=pre.get("docs_dir", "documents"),
        md_dir=pre.get("md_dir"),
        filtered_dir=pre.get("filtered_dir"),
        compressed_dir=pre.get("compressed_dir"),
        output_base=pre.get("output_base"),
        datatrove_dir=pre.get("datatrove_dir"),
    )

    # 从 llm 配置中提取 compress 需要的参数
    llm_cfg = cfg.get("llm", {})
    compress_kwargs = dict(cfg.get("compress", {}))
    if llm_cfg.get("api_key") and "api_key" not in compress_kwargs:
        compress_kwargs["api_key"] = llm_cfg["api_key"]
    if llm_cfg.get("base_url") and "base_url" not in compress_kwargs:
        compress_kwargs["base_url"] = llm_cfg["base_url"]
    if llm_cfg.get("model") and "model" not in compress_kwargs:
        compress_kwargs["model"] = llm_cfg["model"]

    return pipeline.run_all(
        convert_kwargs=cfg.get("convert", {}),
        quality_kwargs=cfg.get("quality", {}),
        compress_kwargs=compress_kwargs,
    )
