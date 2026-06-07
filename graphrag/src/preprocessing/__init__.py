"""文档预处理模块：原始文档 → Markdown → 质量过滤 → LLM 压缩 → GraphRAG"""

from graphrag.src.preprocessing.convert_to_md import (
    convert_file,
    batch_convert,
    validate_md_files,
    reconvert_bad_files,
)
from graphrag.src.preprocessing.quality_pipeline import (
    MarkdownReader,
    ChineseQualityScorer,
    ChineseQualityFilter,
    QualityStatsCollector,
    MarkdownWriter,
    build_pipeline,
    QualityPipeline,
)
from graphrag.src.preprocessing.llm_compress import (
    estimate_tokens,
    calculate_cost,
    compress_with_llm,
    process_file,
    generate_compare_report,
    LLMCompressor,
)
from graphrag.src.preprocessing.dataset_generator import (
    DatasetGenerator,
    generate_qa_with_llm,
)
from graphrag.src.preprocessing.pipeline import (
    PreprocessingPipeline,
    run_preprocessing,
)

__all__ = [
    # convert_to_md
    "convert_file", "batch_convert", "validate_md_files", "reconvert_bad_files",
    # quality_pipeline
    "MarkdownReader", "ChineseQualityScorer", "ChineseQualityFilter",
    "QualityStatsCollector", "MarkdownWriter", "build_pipeline", "QualityPipeline",
    # llm_compress
    "estimate_tokens", "calculate_cost", "compress_with_llm",
    "process_file", "generate_compare_report", "LLMCompressor",
    # dataset_generator
    "DatasetGenerator", "generate_qa_with_llm",
    # pipeline
    "PreprocessingPipeline", "run_preprocessing",
]
