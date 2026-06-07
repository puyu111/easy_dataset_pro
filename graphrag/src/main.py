import logging
import pickle
from pathlib import Path
from typing import Any, Optional

from graphrag.src.config.settings import Settings, load_config
from graphrag.src.embeddings.embedder import BaseEmbedder, EmbedderFactory
from graphrag.src.similarity.base import BaseSimilarity
from graphrag.src.similarity.factory import SimilarityFactory
from graphrag.src.chunking.semantic_chunker import SemanticChunker, TextChunk
from graphrag.src.graph.builder import GraphBuilder
from graphrag.src.graph.retriever import GraphRetriever
from graphrag.src.generation.generator import Generator
from graphrag.src.rerank.base import BaseReranker
from graphrag.src.rerank.factory import RerankerFactory

logger = logging.getLogger(__name__)


class GraphRAG:
    """GraphRAG 完整管线的编排器。"""

    def __init__(
        self,
        config_path: str = "",
        settings: Optional[Settings] = None,
        embedder: Optional[BaseEmbedder] = None,
        similarity: Optional[BaseSimilarity] = None,
        reranker: Optional[BaseReranker] = None,
        llm_client: Any = None,
    ):
        self.settings = settings or load_config(config_path)
        self.embedder = embedder or EmbedderFactory.create(self.settings)
        self.similarity = similarity or SimilarityFactory.create(self.settings)
        self.reranker = reranker or (
            RerankerFactory.create(self.settings) if self.settings.rerank.enabled else None
        )
        self.llm_client = llm_client or self._create_llm_client()

        # 创建 Neo4j driver（如果启用）
        self.driver = self._create_neo4j_driver()

        self.chunker: SemanticChunker = SemanticChunker(
            embedder=self.embedder,
            similarity=self.similarity,
            config=self.settings.chunking,
        )
        self.graph_builder: GraphBuilder = GraphBuilder(
            config=self.settings.graph,
            llm_client=self.llm_client,
            driver=self.driver,
        )
        self.generator: Generator = Generator(
            config=self.settings.llm,
            client=self.llm_client,
        )
        self.retriever: Optional[GraphRetriever] = None

        self.documents: dict[str, str] = {}
        self.chunks: list[TextChunk] = []

        logger.info(
            "GraphRAG 初始化完成: embedder=%s, similarity=%s, llm=%s, rerank=%s, neo4j=%s",
            self.settings.embeddings.provider,
            self.settings.similarity.strategy,
            self.settings.llm.provider,
            self.settings.rerank.provider if self.settings.rerank.enabled else "disabled",
            "enabled" if self.driver is not None else "disabled",
        )

    def _create_neo4j_driver(self) -> Any:
        """根据配置创建 Neo4j driver。"""
        neo4j_config = self.settings.neo4j
        if not neo4j_config.enabled:
            logger.info("Neo4j 未启用，使用内存模式")
            return None
        try:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(
                neo4j_config.uri,
                auth=(neo4j_config.user, neo4j_config.password),
            )
            # 验证连接
            driver.verify_connectivity()
            logger.info("Neo4j 连接成功: %s", neo4j_config.uri)
            return driver
        except Exception as e:
            logger.warning("Neo4j 连接失败，将以内存模式运行: %s", e)
            return None

    def close(self) -> None:
        """关闭 Neo4j driver。"""
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def _create_llm_client(self) -> Any:
        """根据配置创建 LLM 客户端。"""
        if self.settings.llm.provider == "openai":
            from openai import OpenAI

            client_kwargs = {"api_key": self.settings.llm.api_key or None}
            if self.settings.llm.base_url:
                client_kwargs["base_url"] = self.settings.llm.base_url
            return OpenAI(**client_kwargs)
        logger.warning("未知的 LLM 提供者 '%s'；将在没有 LLM 的情况下运行", self.settings.llm.provider)
        return None

    def load_documents(self, documents: dict[str, str]) -> None:
        """加载文档，格式为 {来源: 文本}。"""
        self.documents.update(documents)
        logger.info("已加载 %d 篇文档", len(documents))

    def load_file(self, path: str, source: Optional[str] = None) -> None:
        """加载单个文本文件作为文档。"""
        with open(path) as f:
            text = f.read()
        name = source or path
        self.documents[name] = text
        logger.info("已加载文档 '%s'（%d 字符）", name, len(text))

    def chunk_documents(
        self,
        output_dir: str | None = None,
        resume: bool = True,
    ) -> list[TextChunk]:
        """对已加载的所有文档进行语义切分。

        Args:
            output_dir: 切片结果保存目录，每篇文档一个 JSONL 文件。
                        设置后支持断点续传，每个切片结果可查看。
            resume: 是否跳过已切分的文档（需配合 output_dir 使用）。
        """
        self.chunks = []
        save_dir = Path(output_dir) if output_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        total_docs = len(self.documents)
        skipped = 0

        for doc_idx, (source, text) in enumerate(self.documents.items(), 1):
            safe_name = source.replace("/", "_").replace("\\", "_")
            chunk_file = save_dir / f"{safe_name}.jsonl" if save_dir else None

            # 断点续传：跳过已切分的文档
            if resume and chunk_file and chunk_file.exists():
                loaded = self._load_chunks_from_file(chunk_file)
                if loaded:
                    self.chunks.extend(loaded)
                    skipped += 1
                    logger.info("[%d/%d] 跳过已切分: %s (%d 块)", doc_idx, total_docs, source, len(loaded))
                    continue

            logger.info("[%d/%d] 正在切分: %s (%d 字符)", doc_idx, total_docs, source, len(text))
            chunks = self.chunker.chunk(text, source=source)

            # 逐块打印
            for ci, chunk in enumerate(chunks):
                preview = chunk.text[:80].replace("\n", " ")
                logger.info("  块 %d/%d [%d-%d]: %s...", ci + 1, len(chunks), chunk.start_idx, chunk.end_idx, preview)

            # 保存切片结果
            if chunk_file:
                self._save_chunks_to_file(chunks, chunk_file)

            self.chunks.extend(chunks)
            logger.info("  -> %s: %d 个块", source, len(chunks))

        logger.info("切分完成，共 %d 个块（跳过 %d 篇已切分文档）", len(self.chunks), skipped)
        return self.chunks

    @staticmethod
    def _save_chunks_to_file(chunks: list[TextChunk], path: Path) -> None:
        """将切片结果保存为 JSONL 文件。"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            for c in chunks:
                record = {
                    "text": c.text,
                    "source": c.source,
                    "start_idx": c.start_idx,
                    "end_idx": c.end_idx,
                    "chunk_id": c.chunk_id,
                    "embedding": c.embedding.tolist() if c.embedding is not None else None,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_chunks_from_file(path: Path) -> list[TextChunk]:
        """从 JSONL 文件加载切片结果。"""
        import json
        import numpy as np
        chunks = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    emb = np.array(rec["embedding"], dtype=np.float32) if rec.get("embedding") else None
                    chunks.append(TextChunk(
                        text=rec["text"],
                        source=rec["source"],
                        start_idx=rec["start_idx"],
                        end_idx=rec["end_idx"],
                        chunk_id=rec.get("chunk_id", ""),
                        embedding=emb,
                    ))
        except Exception as e:
            logger.warning("加载切片文件失败 %s: %s", path, e)
            return []
        return chunks

    def build_graph(self, resume: bool = False) -> None:
        """根据所有文本块构建知识图谱。"""
        if not self.chunks:
            logger.warning("没有可用的文本块；请先执行 chunk_documents()")
            return
        self.graph_builder.build(self.chunks, resume=resume)
        self.retriever = GraphRetriever(
            driver=self.driver,
            communities=self.graph_builder.communities,
            config=self.settings.retrieval,
        )
        node_count, edge_count = self._get_graph_stats()
        logger.info(
            "图谱已就绪: %d 个节点, %d 条边",
            node_count,
            edge_count,
        )

    def _get_graph_stats(self) -> tuple[int, int]:
        """返回 (节点数, 边数)。"""
        if self.driver is not None:
            try:
                with self.driver.session() as session:
                    n = session.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
                    e = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS c").single()["c"]
                    return n, e
            except Exception:
                pass
        return (
            len(self.graph_builder.communities) if self.graph_builder.communities else 0,
            0,
        )

    def query(self, question: str, top_k_chunks: int | None = None, chain_of_thought: bool = False) -> str:
        """执行完整的检索+生成管线来回答问题。

        Args:
            question: 用户问题。
            top_k_chunks: 检索的文本块数量。
            chain_of_thought: 是否启用思维链（CoT）。
        """
        if self.retriever is None:
            logger.warning("图谱尚未构建，现在开始构建")
            self.build_graph()

        # 确定检索和重排序的参数
        if top_k_chunks is not None:
            final_top_k = top_k_chunks
            retrieve_top_k = top_k_chunks
        elif self.settings.rerank.enabled:
            final_top_k = self.settings.rerank.top_k
            retrieve_top_k = self.settings.rerank.retrieve_k
        else:
            final_top_k = 5
            retrieve_top_k = 5

        # 从问题中提取查询关键词
        query_terms = self._extract_query_terms(question)

        # 图谱检索
        graph_context = self.retriever.retrieve_context(query_terms)  # type: ignore

        # 文本块检索（初始检索，数量可能多于最终需要的）
        relevant_chunks = self._retrieve_chunks(question, retrieve_top_k)

        # 重排序（如果启用）
        if self.reranker is not None and len(relevant_chunks) > 1:
            relevant_chunks = self.reranker.rerank(question, relevant_chunks, final_top_k)
            logger.info("重排序后保留 %d 个文本块", len(relevant_chunks))
        elif len(relevant_chunks) > final_top_k:
            relevant_chunks = relevant_chunks[:final_top_k]

        # 生成答案
        answer = self.generator.generate(
            question=question,
            graph_context=graph_context,
            text_chunks=relevant_chunks,
            chain_of_thought=chain_of_thought,
        )
        return answer

    def _extract_query_terms(self, question: str) -> list[str]:
        """从问题中提取关键词，用于图谱检索。"""
        import re

        # 去除常见停用词和标点
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "who",
            "where", "when", "why", "how", "does", "do", "did", "can",
            "could", "will", "would", "should", "has", "have", "had",
            "this", "that", "these", "those", "of", "in", "on", "at",
            "to", "for", "with", "by", "about", "as", "into", "through",
            "during", "before", "after", "above", "below", "between",
            "and", "or", "but", "not", "no", "be", "been", "being",
        }
        cleaned = re.sub(r"[^\w\s]", " ", question.lower())
        terms = [
            t.strip() for t in cleaned.split()
            if t.strip() and t.strip() not in stop_words and len(t.strip()) > 2
        ]
        return terms

    def _retrieve_chunks(self, question: str, top_k: int) -> list[dict]:
        """通过 embedding 相似度检索 top-k 文本块。"""
        if not self.chunks:
            return []

        query_emb = self.embedder.embed_one(question)
        scored: list[tuple[float, TextChunk]] = []
        for chunk in self.chunks:
            if chunk.embedding is not None:
                sim = self.similarity.compute(query_emb, chunk.embedding)
                scored.append((sim, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "score": float(score),
            }
            for score, chunk in scored[:top_k]
        ]

    def export_graph(self, path: str) -> None:
        """将知识图谱导出为 GEXF 格式。"""
        self.graph_builder.export_gexf(path)

    def get_statistics(self) -> dict:
        """返回管线统计信息。"""
        node_count, edge_count = self._get_graph_stats()
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "graph_nodes": node_count,
            "graph_edges": edge_count,
            "communities": len(self.graph_builder.communities),
            "rerank": self.settings.rerank.provider if self.settings.rerank.enabled else "disabled",
        }

    def save(self, path: str) -> None:
        """将构建好的 RAG 状态保存到磁盘（文档、文本块、社区）。

        保存后可通过 GraphRAG.load() 加载，无需重新构建。
        """
        state = {
            "documents": self.documents,
            "chunks": self.chunks,
            "communities": self.graph_builder.communities,
        }
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("RAG 状态已保存到: %s", save_path)

    @classmethod
    def load(cls, path: str, config_path: str = "") -> "GraphRAG":
        """从磁盘加载已构建的 RAG 状态，跳过文档加载和图谱构建。

        Args:
            path: save() 保存的文件路径。
            config_path: 配置文件路径，用于初始化 embedder、LLM 等组件。
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        rag = cls(config_path=config_path)
        rag.documents = state["documents"]
        rag.chunks = state["chunks"]
        rag.graph_builder.communities = state["communities"]

        # 重建 retriever（需要 communities 和 config）
        rag.retriever = GraphRetriever(
            driver=rag.driver,
            communities=rag.graph_builder.communities,
            config=rag.settings.retrieval,
        )

        logger.info(
            "RAG 状态已加载: %d 篇文档, %d 个文本块, %d 个社区",
            len(rag.documents), len(rag.chunks), len(rag.graph_builder.communities),
        )
        return rag
