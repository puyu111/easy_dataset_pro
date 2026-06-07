"""重排序器单元测试。"""

import pytest

from graphrag.src.rerank.base import BaseReranker
from graphrag.src.rerank.cross_encoder import CrossEncoderReranker
from graphrag.src.rerank.factory import RerankerFactory
from graphrag.src.config.settings import RerankConfig


class DummyReranker(BaseReranker):
    """用于测试的虚拟重排序器。"""

    def rerank(self, query, documents, top_k):
        # 模拟重排序：按文本长度降序排列
        ranked = sorted(documents, key=lambda d: len(d["text"]), reverse=True)
        for i, doc in enumerate(ranked):
            doc["rerank_score"] = float(len(ranked) - i)
        return ranked[:top_k]


def test_base_reranker_is_abstract():
    """验证 BaseReranker 是抽象类。"""
    import inspect
    assert inspect.isabstract(BaseReranker)


def test_dummy_reranker():
    reranker = DummyReranker()
    docs = [
        {"text": "short", "score": 0.9},
        {"text": "a longer document here", "score": 0.8},
        {"text": "the longest document of all three", "score": 0.7},
    ]
    result = reranker.rerank("test query", docs, top_k=2)
    assert len(result) == 2
    assert result[0]["text"] == "the longest document of all three"
    assert result[1]["text"] == "a longer document here"
    assert "rerank_score" in result[0]


def test_dummy_reranker_empty():
    reranker = DummyReranker()
    result = reranker.rerank("test", [], top_k=5)
    assert result == []


def test_reranker_factory_cross_encoder_disabled():
    """验证禁用时不创建重排序器。"""
    config = RerankConfig(enabled=False)
    assert config.enabled is False


def test_reranker_factory_unknown_provider():
    """验证未知提供者抛出异常。"""
    config = RerankConfig(enabled=True, provider="unknown")
    with pytest.raises(ValueError, match="未知的重排序提供者"):
        RerankerFactory.create(config)


@pytest.mark.skip(reason="需要下载 CrossEncoder 模型，仅用于集成测试")
def test_reranker_factory_cross_encoder_config():
    """验证 CrossEncoder 配置创建（需要下载模型）。"""
    config = RerankConfig(
        enabled=True,
        provider="cross-encoder",
        model="BAAI/bge-reranker-v2-m3",
    )
    reranker = RerankerFactory.create(config)
    assert isinstance(reranker, CrossEncoderReranker)


def test_dummy_reranker_with_scores():
    """验证重排序后分数正确更新。"""
    reranker = DummyReranker()
    docs = [
        {"text": "短文本", "score": 0.5},
        {"text": "这是一段较长的文本用于测试重排序", "score": 0.4},
    ]
    result = reranker.rerank("查询", docs, top_k=2)
    # 较长的文本应该排在前面
    assert result[0]["text"] == "这是一段较长的文本用于测试重排序"
    assert result[0]["rerank_score"] > result[1]["rerank_score"]
