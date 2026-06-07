import abc

import numpy as np


class BaseSimilarity(abc.ABC):
    """相似度计算的抽象基类（策略模式）。"""

    @abc.abstractmethod
    def compute(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算两个 1D 向量之间的相似度。"""
        ...

    def compute_batch(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        """批量计算多对向量的相似度。"""
        return [self.compute(a, b) for a, b in pairs]

    def compute_matrix(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算两组向量之间的相似度矩阵。

        返回一个 (n, m) 矩阵，其中元素 (i, j) = sim(X[i], Y[j])。
        """
        n, m = X.shape[0], Y.shape[0]
        result = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            for j in range(m):
                result[i, j] = self.compute(X[i], Y[j])
        return result
