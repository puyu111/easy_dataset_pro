import logging
import time
from typing import Any, Optional

from .base import BaseReranker

logger = logging.getLogger(__name__)


class OpenAIReranker(BaseReranker):
    """基于 OpenAI 兼容 API 的重排序器。

    通过 POST {base_url}/rerank 接口对文档进行重排序。
    兼容 DashScope、Cohere、Jina 等服务商。
    """

    def __init__(
        self,
        model: str = "rerank-v1",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    # qwen3-rerank 限制
    MAX_DOCS_PER_REQUEST = 500
    MAX_TOKENS_PER_REQUEST = 120_000
    MAX_CHARS_PER_DOC = 4000  # 约等于单条 4000 token 上限

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["POST"],
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        url = f"{self.base_url.rstrip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 截断超长文档
        doc_texts = [doc["text"][:self.MAX_CHARS_PER_DOC] for doc in documents]
        logger.debug("重排序请求: url=%s, model=%s, docs=%d, 总字符=%d",
                      url, self.model, len(doc_texts), sum(len(t) for t in doc_texts))

        # 分批：按文档数和 token 估算分组
        batches = self._split_batches(doc_texts, query)
        all_results: list[dict] = []

        try:
            for batch_indices, batch_texts in batches:
                try:
                    payload = {
                        "model": self.model,
                        "query": query,
                        "documents": batch_texts,
                        "top_n": len(batch_texts),
                    }
                    # 重试最多 3 次（含延迟）
                    data = None
                    for attempt in range(3):
                        resp = session.post(url, headers=headers, json=payload, timeout=60)
                        if resp.status_code != 200:
                            logger.warning("重排序 API 返回 %d: %s", resp.status_code, resp.text[:300])
                            time.sleep(2 * (attempt + 1))
                            continue
                        if not resp.text.strip():
                            logger.warning("重排序 API 返回空响应，重试 %d/3", attempt + 1)
                            time.sleep(2 * (attempt + 1))
                            continue
                        data = resp.json()
                        break
                    if data is None:
                        logger.warning("重排序批次 3 次重试均失败，跳过")
                        continue

                    results = data.get("results") or data.get("output", {}).get("results", [])
                    for item in results:
                        idx = item.get("index")
                        if idx is not None and 0 <= idx < len(batch_indices):
                            all_results.append({
                                "orig_idx": batch_indices[idx],
                                "relevance_score": item.get("relevance_score", 0.0),
                            })
                except Exception as e:
                    logger.warning("重排序批次失败 (%s)，跳过该批次", e)
                    continue
                # 批次间延迟，避免限流
                if len(batches) > 1:
                    time.sleep(1)

            if not all_results:
                logger.warning("所有重排序批次均失败，使用原始排序")
                return documents[:top_k]

            # 按分数排序，取 top_k
            all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
            ranked: list[dict[str, Any]] = []
            for item in all_results[:top_k]:
                doc = documents[item["orig_idx"]]
                doc["rerank_score"] = float(item["relevance_score"])
                ranked.append(doc)

            logger.info("重排序完成: %d -> %d 个结果（%d 批次）", len(documents), len(ranked), len(batches))
            return ranked

        except Exception as e:
            logger.warning("重排序 API 请求失败 (%s)，使用原始排序", e)
            return documents[:top_k]
        finally:
            session.close()

    def _split_batches(
        self, doc_texts: list[str], query: str
    ) -> list[tuple[list[int], list[str]]]:
        """将文档分成多批，每批不超过文档数和 token 限制。"""
        # 估算 token 数（中文约 1.5 token/字符）
        query_tokens = len(query) * 2
        batches: list[tuple[list[int], list[str]]] = []
        cur_indices: list[int] = []
        cur_texts: list[str] = []
        cur_tokens = query_tokens

        for i, text in enumerate(doc_texts):
            doc_tokens = len(text) * 2  # 粗略估算
            # 超过限制则开新批
            if cur_indices and (
                len(cur_indices) >= self.MAX_DOCS_PER_REQUEST
                or cur_tokens + doc_tokens > self.MAX_TOKENS_PER_REQUEST
            ):
                batches.append((cur_indices, cur_texts))
                cur_indices = []
                cur_texts = []
                cur_tokens = query_tokens
            cur_indices.append(i)
            cur_texts.append(text)
            cur_tokens += doc_tokens

        if cur_indices:
            batches.append((cur_indices, cur_texts))

        return batches
