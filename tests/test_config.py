from graphrag.src.config.settings import load_config, Settings


def test_load_config():
    settings = load_config("graphrag/config/config.yaml")
    assert settings.similarity.strategy == "cosine"
    assert settings.chunking.similarity_threshold == 0.75
    assert settings.chunking.min_chunk_size == 50
    assert settings.embeddings.provider == "openai"
    assert settings.embeddings.openai_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert settings.graph.extraction_model == "openai"
    assert settings.graph.community_detection.resolution == 1.0
    assert settings.retrieval.max_entities == 10
    assert settings.retrieval.neighbor_hops == 2
    assert settings.llm.provider == "openai"
    assert settings.llm.base_url == "https://api.deepseek.com"
    assert settings.llm.temperature == 0.3
    assert settings.rerank.enabled is True
    assert settings.rerank.provider == "openai"
    assert settings.rerank.top_k == 5


def test_settings_defaults():
    settings = Settings()
    assert settings.similarity.strategy == "cosine"
    assert settings.chunking.min_chunk_size == 50
    assert settings.embeddings.provider == "sentence-transformers"
    assert settings.rerank.enabled is False
    assert settings.rerank.provider == "cross-encoder"


def test_settings_from_dict():
    data = {
        "similarity": {"strategy": "dot_product"},
        "chunking": {"min_chunk_size": 100},
        "embeddings": {"provider": "openai"},
        "graph": {"extraction_model": "openai"},
        "retrieval": {"max_entities": 20},
        "llm": {"temperature": 0.5},
        "rerank": {"enabled": True, "provider": "openai", "top_k": 10},
    }
    settings = Settings.from_dict(data)
    assert settings.similarity.strategy == "dot_product"
    assert settings.chunking.min_chunk_size == 100
    assert settings.embeddings.provider == "openai"
    assert settings.retrieval.max_entities == 20
    assert settings.llm.temperature == 0.5
    assert settings.rerank.enabled is True
    assert settings.rerank.provider == "openai"
    assert settings.rerank.top_k == 10
