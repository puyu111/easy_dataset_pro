import numpy as np

from graphrag.src.chunking.semantic_chunker import SemanticChunker, TextChunk
from graphrag.src.config.settings import ChunkingConfig
from graphrag.src.similarity.cosine import CosineSimilarity


class DummyEmbedder:
    """Embedder that returns deterministic embeddings for testing."""

    def __init__(self):
        self.counter = 0

    def embed(self, texts):
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 4)
        results = []
        for text in texts:
            # Create a deterministic embedding based on text length
            emb = np.full(4, 0.1, dtype=np.float32)
            emb[0] = len(text) / 100.0
            emb[1] = hash(text) % 1000 / 1000.0
            results.append(emb)
        return np.array(results)

    def embed_one(self, text):
        return self.embed([text])[0]

    def close(self):
        pass


def test_sentence_splitting_basic():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
        config=ChunkingConfig(similarity_threshold=0.5),
    )
    sentences = chunker._split_sentences("Hello world. This is a test. How are you?")
    assert len(sentences) == 3
    assert "Hello world" in sentences[0]
    assert "This is a test" in sentences[1]
    assert "How are you" in sentences[2]


def test_sentence_splitting_single():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
    )
    sentences = chunker._split_sentences("Just one sentence here.")
    assert len(sentences) == 1
    assert "Just one sentence here" in sentences[0]


def test_sentence_splitting_no_punctuation():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
    )
    sentences = chunker._split_sentences("No punctuation at all in this text")
    assert len(sentences) == 1


def test_chunk_creation():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
        config=ChunkingConfig(similarity_threshold=0.5),
    )
    text = "Apple is a technology company. It makes iPhones. Google is a search engine company. It makes Android."
    chunks = chunker.chunk(text, source="test_doc")
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.source == "test_doc"
        assert chunk.start_idx >= 0
        assert chunk.end_idx > chunk.start_idx
        assert chunk.embedding is not None


def test_chunk_empty_text():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
    )
    chunks = chunker.chunk("", source="empty")
    assert len(chunks) == 0


def test_chunk_whitespace_text():
    chunker = SemanticChunker(
        embedder=DummyEmbedder(),
        similarity=CosineSimilarity(),
    )
    chunks = chunker.chunk("   \n\n   ", source="whitespace")
    assert len(chunks) == 0
