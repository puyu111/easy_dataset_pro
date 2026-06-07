import abc
from typing import Any


class BaseReranker(abc.ABC):
    """重排序器的抽象基类（策略模式）。"""

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """对检索结果进行重排序，返回重排序后的 top_k 个文档。

        参数:
            query: 原始查询文本
            documents: 文档列表，每项至少包含 "text" 和 "score" 键
            top_k: 返回的最终文档数量

        返回:
            重排序后的文档列表（按新分数降序），每项新增 "rerank_score" 键
        """
        ...
