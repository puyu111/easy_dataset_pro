import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from graphrag.src.config.settings import ChunkingConfig, Settings
from graphrag.src.embeddings.embedder import BaseEmbedder
from graphrag.src.similarity.base import BaseSimilarity

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    text: str
    source: str
    start_idx: int
    end_idx: int
    embedding: Optional[np.ndarray] = None
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            self.chunk_id = f"{self.source}:{self.start_idx}-{self.end_idx}"


_SENTENCE_PATTERN = re.compile(
    r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|\!|\n)\s+(?=[A-Z\"\'\(])",
)


class SemanticChunker:
    """利用 embedding 相似度将文本切分为语义连贯的块。"""

    def __init__(
        self,
        embedder: BaseEmbedder,
        similarity: BaseSimilarity,
        config: ChunkingConfig | None = None,
    ):
        self.embedder = embedder
        self.similarity = similarity
        self.config = config or ChunkingConfig()

    def _split_sentences(self, text: str) -> list[str]:
        """用正则表达式将文本切分为句子。"""
        raw = _SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in raw if s.strip()]
        return sentences if sentences else [text.strip()]

    def chunk(self, text: str, source: str = "") -> list[TextChunk]:
        """根据 embedding 相似度将文本切分为语义块。"""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        logger.info("正在切分 '%s'，共 %d 个句子", source, len(sentences))

        sentence_embeddings = self.embedder.embed(sentences)

        # 计算相邻句子之间的相似度
        similarities: list[float] = []
        for i in range(len(sentences) - 1):
            sim = self.similarity.compute(sentence_embeddings[i], sentence_embeddings[i + 1])
            similarities.append(sim)

        # 找到相似度低于阈值的切分点
        threshold = self.config.similarity_threshold
        split_indices = [0]
        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_indices.append(i + 1)
        split_indices.append(len(sentences))

        # 合并 chunk 以满足最小/最大大小约束
        chunks = self._merge_chunks(sentences, split_indices)

        # 为每个块计算 embedding
        chunk_texts = [c.text for c in chunks]
        chunk_embeddings = self.embedder.embed(chunk_texts)

        result: list[TextChunk] = []
        char_offset = 0
        for (c, emb) in zip(chunks, chunk_embeddings):
            start = text.find(c.text[:20], char_offset)
            if start == -1:
                start = char_offset
            end = start + len(c.text)
            result.append(
                TextChunk(
                    text=c.text,
                    source=source,
                    start_idx=start,
                    end_idx=end,
                    embedding=emb,
                )
            )
            char_offset = end

        logger.info("从 '%s' 创建了 %d 个块", source, len(result))
        return result

    def _merge_chunks(
        self,
        sentences: list[str],
        split_indices: list[int],
    ) -> list["_Chunk"]:
        """合并句子组以满足最小/最大块大小约束。"""
        raw_chunks: list[str] = []
        for i in range(len(split_indices) - 1):
            group = sentences[split_indices[i] : split_indices[i + 1]]
            raw_chunks.append(" ".join(group))

        min_size = self.config.min_chunk_size
        max_size = self.config.max_chunk_size

        merged: list[str] = []
        buffer = ""
        for chunk in raw_chunks:
            if not buffer:
                buffer = chunk
            elif len(buffer) + len(chunk) <= max_size:
                buffer += " " + chunk
            else:
                if len(buffer) < min_size and merged:
                    # 合并到前一个块
                    merged[-1] += " " + buffer
                else:
                    merged.append(buffer)
                buffer = chunk

        if buffer:
            if len(buffer) < min_size and merged:
                merged[-1] += " " + buffer
            else:
                merged.append(buffer)

        # 包装为下游使用的简单对象
        class _Chunk:
            def __init__(self, text: str):
                self.text = text

        return [_Chunk(t) for t in merged]
