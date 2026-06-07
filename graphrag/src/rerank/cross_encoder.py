import logging
from typing import Any

from .base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """基于 sentence-transformers CrossEncoder 的本地重排序器。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        logger.info("CrossEncoder 重排序器已加载: %s", model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.model.predict(pairs, show_progress_bar=False)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        logger.info("CrossEncoder 重排序完成: %d -> %d 个结果", len(documents), top_k)
        return ranked[:top_k]
