import abc
import logging
from typing import Optional

import numpy as np

from graphrag.src.config.settings import EmbeddingConfig, Settings

logger = logging.getLogger(__name__)


class BaseEmbedder(abc.ABC):
    """嵌入提供者的抽象基类。"""

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """将文本列表嵌入为二维 numpy 数组 (文本数 x 嵌入维度)。"""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @abc.abstractmethod
    def close(self) -> None:
        """释放嵌入器持有的所有资源。"""
        ...


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None, base_url: Optional[str] = None, batch_size: int = 10):
        from openai import OpenAI

        self.model = model
        self.batch_size = min(batch_size, 10)  # API 限制单次最多 10 条
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        # 记录非空文本的索引，空文本用零向量占位
        non_empty_indices = [i for i, t in enumerate(texts) if t.strip()]
        if not non_empty_indices:
            return np.zeros((len(texts), 1), dtype=np.float32)

        non_empty_texts = [texts[i] for i in non_empty_indices]

        # 截断超长文本（API 限制 8192 tokens，粗略按字符截断）
        max_chars = 6000  # 留余量，中文约 1.5 token/字
        truncated = [t[:max_chars] if len(t) > max_chars else t for t in non_empty_texts]

        all_embeddings = []
        for i in range(0, len(truncated), self.batch_size):
            batch = truncated[i:i + self.batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            all_embeddings.extend(item.embedding for item in response.data)

        # 组装结果：非空位置填入 embedding，空位置填零向量
        dim = len(all_embeddings[0])
        result = np.zeros((len(texts), dim), dtype=np.float32)
        for idx, emb in zip(non_empty_indices, all_embeddings):
            result[idx] = emb
        return result

    def close(self) -> None:
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False).astype(np.float32)

    def close(self) -> None:
        pass


class EmbedderFactory:
    @staticmethod
    def create(config: EmbeddingConfig | Settings) -> BaseEmbedder:
        if isinstance(config, Settings):
            config = config.embeddings
        provider = config.provider
        logger.info("创建嵌入器: provider='%s', model='%s'", provider, config.model)
        if provider == "openai":
            return OpenAIEmbedder(
                model=config.openai_model,
                api_key=config.openai_api_key or None,
                base_url=config.openai_base_url or None,
                batch_size=config.batch_size,
            )
        elif provider == "sentence-transformers":
            return SentenceTransformerEmbedder(model_name=config.model)
        else:
            raise ValueError(f"未知的嵌入提供者: {provider}")
