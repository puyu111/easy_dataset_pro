import logging

from graphrag.src.config.settings import Settings, SimilarityConfig
from .base import BaseSimilarity
from .cosine import CosineSimilarity
from .dot_product import DotProductSimilarity

logger = logging.getLogger(__name__)


class SimilarityFactory:
    @staticmethod
    def create(config: SimilarityConfig | Settings) -> BaseSimilarity:
        if isinstance(config, Settings):
            config = config.similarity
        strategy = config.strategy
        logger.info("创建相似度策略: '%s'", strategy)
        if strategy == "cosine":
            return CosineSimilarity()
        elif strategy == "dot_product":
            return DotProductSimilarity()
        else:
            raise ValueError(f"未知的相似度策略: {strategy}")
