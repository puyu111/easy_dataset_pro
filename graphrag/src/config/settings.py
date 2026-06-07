import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class LoggingConfig:
    level: str = "INFO"
    directory: str = "logs"
    filename: str = "graphrag.log"
    format_console: str = "[%(levelname)s] %(message)s"
    format_file: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    console_enabled: bool = True
    file_enabled: bool = True


@dataclass
class DocumentsConfig:
    directory: str = ""


@dataclass
class SimilarityConfig:
    strategy: str = "cosine"


@dataclass
class ChunkingConfig:
    similarity_threshold: float = 0.75
    min_chunk_size: int = 50
    max_chunk_size: int = 2000


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    openai_base_url: str = ""  # 例如: https://api.openai.com/v1
    batch_size: int = 32


@dataclass
class CommunityDetectionConfig:
    resolution: float = 1.0


@dataclass
class GraphConfig:
    extraction_model: str = "openai"
    extraction_model_name: str = "gpt-4o-mini"
    max_entities_per_chunk: int = 20
    max_relations_per_chunk: int = 40
    community_detection: CommunityDetectionConfig = field(default_factory=CommunityDetectionConfig)


@dataclass
class RetrievalConfig:
    max_entities: int = 10
    neighbor_hops: int = 2
    max_community_results: int = 5


@dataclass
class RerankConfig:
    enabled: bool = False
    provider: str = "cross-encoder"  # "cross-encoder" or "openai"
    model: str = "BAAI/bge-reranker-v2-m3"
    api_key: str = ""
    base_url: str = ""
    top_k: int = 5       # 重排序后保留的结果数
    retrieve_k: int = 20  # 重排序前初始检索的结果数


@dataclass
class Neo4jConfig:
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"


@dataclass
class BatchQueryConfig:
    """批量问答配置。"""
    dataset: str = "output/dataset/dataset_all.jsonl"
    output: str = "output/dataset/answers.jsonl"
    resume: bool = True
    save_interval: int = 10
    state: str = "output/rag_state.pkl"
    chain_of_thought: bool = False  # 是否启用思维链（CoT），让 LLM 先推理再给出最终答案


@dataclass
class PreprocessingConfig:
    """文档预处理配置。"""
    enabled: bool = False
    docs_dir: str = "documents"
    md_dir: str = ""         # 默认为 {docs_dir}_md
    filtered_dir: str = ""   # 默认为 {output_base}/filtered_md
    compressed_dir: str = "" # 默认为 {output_base}/llm_compressed
    output_base: str = ""    # 输出根目录（默认为 docs_dir 的父目录下的 output）
    datatrove_dir: str = ""  # datatrove 源码路径

    # Stage 1: convert
    convert_formats: list = None
    convert_per_page: bool = False
    convert_image_mode: str = "embedded"
    convert_page_breaks: bool = True
    convert_do_ocr: bool = True
    convert_force_ocr: bool = False
    convert_max_pages: int = 50
    convert_resume: bool = True
    convert_device: str = "auto"
    convert_validate: bool = True
    convert_reconvert_bad: bool = False

    # Stage 2: quality
    quality_mode: str = "full"
    quality_min_chars: int = 200
    quality_min_score: float = 0.4

    # Stage 3: compress
    compress_max_chunk_tokens: int = 24000
    compress_max_output_tokens: int = 8192
    compress_concurrency: int = 1
    compress_resume: bool = True
    compress_no_chunk: bool = False

    # Stage 4: dataset generation
    dataset_questions_per_chunk: int = 5
    dataset_max_chunk_tokens: int = 24000
    dataset_concurrency: int = 1
    dataset_no_chunk: bool = True

    def __post_init__(self):
        if self.convert_formats is None:
            self.convert_formats = [".pdf", ".docx", ".pptx", ".html", ".htm"]


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str = ""  # 例如: https://api.openai.com/v1
    temperature: float = 0.3
    max_tokens: int = 2048


@dataclass
class Settings:
    documents: DocumentsConfig = field(default_factory=DocumentsConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    batch: BatchQueryConfig = field(default_factory=BatchQueryConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Settings":
        documents = DocumentsConfig(**data.get("documents", {}))
        similarity = SimilarityConfig(**data.get("similarity", {}))
        chunking = ChunkingConfig(**data.get("chunking", {}))
        embedding_kwargs = data.get("embeddings", {})
        embedding_kwargs["openai_api_key"] = (
            embedding_kwargs.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        embedding_kwargs["openai_base_url"] = (
            embedding_kwargs.get("openai_base_url")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        embeddings = EmbeddingConfig(**embedding_kwargs)
        graph_kwargs = data.get("graph", {})
        cd_config = CommunityDetectionConfig(
            **graph_kwargs.pop("community_detection", {})
        )
        graph = GraphConfig(community_detection=cd_config, **graph_kwargs)
        retrieval = RetrievalConfig(**data.get("retrieval", {}))
        rerank_kwargs = data.get("rerank", {})
        rerank_kwargs["api_key"] = (
            rerank_kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        rerank_kwargs["base_url"] = (
            rerank_kwargs.get("base_url")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        rerank = RerankConfig(**rerank_kwargs)
        llm_kwargs = data.get("llm", {})
        llm_kwargs["api_key"] = (
            llm_kwargs.get("api_key")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        llm_kwargs["base_url"] = (
            llm_kwargs.get("base_url")
            or os.environ.get("OPENAI_BASE_URL", "")
        )
        llm = LLMConfig(**llm_kwargs)
        neo4j_kwargs = data.get("neo4j", {})
        neo4j_kwargs["password"] = (
            neo4j_kwargs.get("password")
            or os.environ.get("NEO4J_PASSWORD", "")
        )
        neo4j = Neo4jConfig(**neo4j_kwargs)
        preprocessing = PreprocessingConfig(**data.get("preprocessing", {}))
        logging_config = LoggingConfig(**data.get("logging", {}))
        batch = BatchQueryConfig(**data.get("batch", {}))
        return cls(
            documents=documents,
            similarity=similarity,
            chunking=chunking,
            embeddings=embeddings,
            graph=graph,
            retrieval=retrieval,
            llm=llm,
            rerank=rerank,
            neo4j=neo4j,
            batch=batch,
            preprocessing=preprocessing,
            logging=logging_config,
        )


def load_config(path: str) -> Settings:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Settings.from_dict(data)
