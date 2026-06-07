import numpy as np

from .base import BaseSimilarity


class CosineSimilarity(BaseSimilarity):
    """余弦相似度: sim(a, b) = (a · b) / (||a|| * ||b||)。"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
