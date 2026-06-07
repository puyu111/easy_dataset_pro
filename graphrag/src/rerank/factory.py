import logging

from graphrag.src.config.settings import RerankConfig, Settings
from .base import BaseReranker
from .cross_encoder import CrossEncoderReranker
from .openai_reranker import OpenAIReranker

logger = logging.getLogger(__name__)


class RerankerFactory:
    @staticmethod
    def create(config: RerankConfig | Settings) -> BaseReranker:
        if isinstance(config, Settings):
            config = config.rerank

        provider = config.provider
        logger.info("创建重排序器: provider='%s', model='%s'", provider, config.model)

        if provider == "cross-encoder":
            return CrossEncoderReranker(model_name=config.model)
        elif provider == "openai":
            return OpenAIReranker(
                model=config.model,
                api_key=config.api_key or None,
                base_url=config.base_url or None,
            )
        else:
            raise ValueError(f"未知的重排序提供者: {provider}")
