import numpy as np

from .base import BaseSimilarity


class DotProductSimilarity(BaseSimilarity):
    """点积相似度: sim(a, b) = a · b。"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))
