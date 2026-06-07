import numpy as np

from graphrag.src.similarity.cosine import CosineSimilarity
from graphrag.src.similarity.dot_product import DotProductSimilarity


def test_cosine_similarity_identical():
    sim = CosineSimilarity()
    a = np.array([1.0, 2.0, 3.0])
    assert abs(sim.compute(a, a) - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    sim = CosineSimilarity()
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert abs(sim.compute(a, b)) < 1e-6


def test_cosine_similarity_opposite():
    sim = CosineSimilarity()
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(sim.compute(a, b) - (-1.0)) < 1e-6


def test_cosine_zero_vector():
    sim = CosineSimilarity()
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert sim.compute(a, b) == 0.0
    assert sim.compute(b, a) == 0.0


def test_cosine_known_value():
    sim = CosineSimilarity()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    result = sim.compute(a, b)
    expected = (4 + 10 + 18) / (np.sqrt(14) * np.sqrt(77))
    assert abs(result - expected) < 1e-6


def test_dot_product():
    sim = DotProductSimilarity()
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    expected = 4.0 + 10.0 + 18.0
    assert abs(sim.compute(a, b) - expected) < 1e-6


def test_dot_product_zero():
    sim = DotProductSimilarity()
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 3.0])
    assert abs(sim.compute(a, b)) < 1e-6


def test_compute_batch():
    sim = CosineSimilarity()
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    c = np.array([1.0, 1.0])
    pairs = [(a, b), (a, a), (b, c)]
    results = sim.compute_batch(pairs)
    assert len(results) == 3
    assert abs(results[0]) < 1e-6  # orthogonal
    assert abs(results[1] - 1.0) < 1e-6  # identical
    assert abs(results[2] - (1.0 / np.sqrt(2))) < 1e-6


def test_compute_matrix():
    sim = CosineSimilarity()
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    Y = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    matrix = sim.compute_matrix(X, Y)
    assert matrix.shape == (2, 3)
    assert abs(matrix[0, 0] - 1.0) < 1e-6  # (1,0) · (1,0) = 1
    assert abs(matrix[0, 1]) < 1e-6  # (1,0) · (0,1) = 0
    assert abs(matrix[1, 1] - 1.0) < 1e-6  # (0,1) · (0,1) = 1
    assert abs(matrix[1, 0]) < 1e-6  # (0,1) · (1,0) = 0
