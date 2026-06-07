import pytest
from unittest.mock import MagicMock

from graphrag.src.graph.builder import Entity, Relation, GraphDocument, GraphBuilder
from graphrag.src.graph.retriever import GraphRetriever
from graphrag.src.chunking.semantic_chunker import TextChunk
from graphrag.src.config.settings import GraphConfig, RetrievalConfig


@pytest.fixture
def mock_driver():
    """Create a mock Neo4j driver with a session context manager."""
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


@pytest.fixture
def mock_session(mock_driver):
    """Return the mock session from mock_driver."""
    return mock_driver.session.return_value.__enter__.return_value


def make_result(records: list[dict]):
    """Create a mock Neo4j result that yields dict records (supports record['key'])."""
    result = MagicMock()
    result.__iter__.return_value = iter(records)
    result.single.return_value = records[0] if records else None
    return result


# --- Dataclass tests (unchanged) ---

def test_entity_dataclass():
    entity = Entity(name="Apple", type="Company", description="Tech company")
    assert entity.name == "Apple"
    assert entity.type == "Company"
    assert entity.description == "Tech company"


def test_relation_dataclass():
    rel = Relation(source="Apple", target="iPhone", relation="produces")
    assert rel.source == "Apple"
    assert rel.target == "iPhone"
    assert rel.relation == "produces"


def test_graph_document():
    doc = GraphDocument(
        entities=[Entity(name="A"), Entity(name="B")],
        relations=[Relation(source="A", target="B", relation="connected")],
    )
    assert len(doc.entities) == 2
    assert len(doc.relations) == 1


# --- Builder tests ---

def test_graph_builder_rule_based():
    """Test rule-based extraction with (entity - type - desc) and [source -> target: relation] patterns."""
    builder = GraphBuilder(config=GraphConfig())
    text = (
        "Some text with (Apple - Company - A tech company) and "
        "(iPhone - Product - A smartphone). "
        "Also [Apple -> iPhone: produces] in this text."
    )
    chunk = TextChunk(text=text, source="test", start_idx=0, end_idx=len(text))
    doc = builder._extract_from_chunk(chunk)
    assert len(doc.entities) == 2
    assert len(doc.relations) == 1
    assert doc.entities[0].name == "Apple"
    assert doc.entities[1].name == "iPhone"
    assert doc.relations[0].source == "Apple"
    assert doc.relations[0].target == "iPhone"


def test_graph_builder_no_patterns():
    """Test that text without patterns yields empty extraction."""
    builder = GraphBuilder(config=GraphConfig())
    chunk = TextChunk(text="Just some random text with no patterns.", source="test", start_idx=0, end_idx=35)
    doc = builder._extract_from_chunk(chunk)
    assert len(doc.entities) == 0
    assert len(doc.relations) == 0


def test_graph_add_to_graph(mock_driver, mock_session):
    """Test that _add_to_graph calls Cypher MERGE for entities and relations."""
    builder = GraphBuilder(config=GraphConfig(), driver=mock_driver)
    doc = GraphDocument(
        entities=[
            Entity(name="A", description="Entity A"),
            Entity(name="B", description="Entity B"),
        ],
        relations=[
            Relation(source="A", target="B", relation="related_to"),
        ],
    )
    builder._add_to_graph(doc, "chunk_1")
    # Should have called session.run for 2 entity MERGEs + 1 relation MERGE
    assert mock_session.run.call_count == 3

    # Verify entity MERGE calls
    entity_calls = [call for call in mock_session.run.call_args_list
                    if "MERGE (n:Entity {name:" in call[0][0] or "MERGE (n:Entity" in call[0][0]]
    assert len(entity_calls) == 2

    # Verify relation MERGE calls
    rel_calls = [call for call in mock_session.run.call_args_list
                 if "MERGE (s:Entity)" in call[0][0] or "MERGE (s)-[r:RELATED]" in call[0][0]]
    assert len(rel_calls) == 1


def test_graph_build_full_pipeline(mock_driver, mock_session):
    """Test the full graph building pipeline with mock driver."""
    # Use a function to return results based on query content
    def run_side_effect(query, **params):
        if "DETACH DELETE" in query or "REMOVE n.community_id" in query:
            return make_result([])
        elif "MERGE" in query:
            return make_result([])
        elif "community_id" in query:
            # GET_ALL_NODES
            return make_result([
                {"name": "Apple", "type": "Company", "description": "Makes iPhones", "community_id": None},
                {"name": "iPhone", "type": "Product", "description": "A smartphone", "community_id": None},
            ])
        elif "RELATED" in query and "source" in query:
            # GET_ALL_RELATIONS
            return make_result([
                {"source": "Apple", "target": "iPhone", "relation": "produces", "description": ""},
            ])
        elif "count(n)" in query or "count(r)" in query:
            return make_result([{"cnt": 2}])
        elif "SET n.community_id" in query or "SET" in query:
            return make_result([])
        return make_result([])

    mock_session.run.side_effect = run_side_effect

    builder = GraphBuilder(config=GraphConfig(), driver=mock_driver)
    chunk = TextChunk(
        text="We found (Apple - Company - Makes iPhones) and [Apple -> iPhone: produces].",
        source="test",
        start_idx=0,
        end_idx=70,
        chunk_id="test:0-70",
    )
    builder.build([chunk])

    # After build, communities should be populated
    assert len(builder.communities) >= 1


def test_graph_builder_no_driver():
    """GraphBuilder without driver should not crash."""
    builder = GraphBuilder(config=GraphConfig())
    chunk = TextChunk(text="(A - T - desc) [A -> B: rel]", source="test", start_idx=0, end_idx=25)
    builder.build([chunk])
    # Without driver, build logs warning but doesn't crash
    assert len(builder.communities) == 0


# --- Retriever tests ---

def test_retriever_entity_match(mock_driver, mock_session):
    """Test entity match with mock Cypher query."""
    mock_session.run.return_value = make_result([
        {"name": "Apple"},
        {"name": "Apple Inc."},
    ])

    retriever = GraphRetriever(
        driver=mock_driver,
        communities=[],
        config=RetrievalConfig(),
    )
    result = retriever.entity_match(["apple"])
    assert "Apple" in result
    assert "Apple Inc." in result
    assert len(result) <= 10


def test_retriever_entity_match_no_match(mock_driver, mock_session):
    """Test entity match returns empty when nothing matches."""
    mock_session.run.return_value = make_result([])

    retriever = GraphRetriever(
        driver=mock_driver,
        communities=[],
        config=RetrievalConfig(),
    )
    result = retriever.entity_match(["nonexistent"])
    assert result == []


def test_retriever_neighbor_expansion(mock_driver, mock_session):
    """Test neighbor expansion returns nodes within N hops."""
    mock_session.run.return_value = make_result([
        {"name": "A"}, {"name": "B"}, {"name": "C"},
    ])

    retriever = GraphRetriever(
        driver=mock_driver,
        communities=[],
        config=RetrievalConfig(),
    )
    expanded = retriever.neighbor_expansion(["A"], hops=2)
    assert "A" in expanded
    assert "B" in expanded
    assert "C" in expanded


def test_retriever_community_retrieval():
    """Test community retrieval uses in-memory communities list."""
    retriever = GraphRetriever(
        driver=None,
        communities=[{"A", "B"}, {"C", "D"}],
        config=RetrievalConfig(),
    )
    matched = retriever.community_retrieval(["A"])
    assert len(matched) == 1
    assert matched[0] == {"A", "B"}


def test_retriever_community_retrieval_multiple():
    """Test community retrieval matches multiple communities."""
    retriever = GraphRetriever(
        driver=None,
        communities=[{"A", "B"}, {"C", "D"}, {"E", "F"}],
        config=RetrievalConfig(max_community_results=5),
    )
    matched = retriever.community_retrieval(["A", "C"])
    assert len(matched) == 2
    assert {"A", "B"} in matched
    assert {"C", "D"} in matched


def test_retriever_multi_hop_path(mock_driver, mock_session):
    """Test shortest path retrieval."""
    mock_session.run.return_value = make_result([
        {"path_names": ["A", "B", "C", "D"]},
    ])

    retriever = GraphRetriever(
        driver=mock_driver,
        communities=[],
        config=RetrievalConfig(),
    )
    paths = retriever.multi_hop_path(["A", "D"])
    assert len(paths) == 1
    assert paths[0] == ["A", "B", "C", "D"]


def test_retriever_retrieve_context(mock_driver, mock_session):
    """Test the full retrieve_context pipeline."""
    # entity_match returns Apple, iPhone
    # neighbor_expansion returns {Apple, iPhone}
    # multi_hop_path returns a path
    def run_side_effect(query, **params):
        if "CONTAINS" in query:
            return make_result([{"name": "Apple"}, {"name": "iPhone"}])
        elif "shortestPath" in query:
            return make_result([{"path_names": ["Apple", "iPhone"]}])
        elif "RELATED" in query and "WHERE" in query:
            return make_result([{"source": "Apple", "target": "iPhone", "relation": "produces"}])
        else:
            return make_result([{"name": "Apple"}, {"name": "iPhone"}])

    mock_session.run.side_effect = run_side_effect

    retriever = GraphRetriever(
        driver=mock_driver,
        communities=[{"Apple", "iPhone"}],
        config=RetrievalConfig(),
    )
    context = retriever.retrieve_context(["apple"])
    assert "entities" in context
    assert "subgraph_nodes" in context
    assert "subgraph_edges" in context
    assert "Apple" in context["entities"]
    assert len(context["subgraph_edges"]) >= 1


def test_empty_graph():
    """Test retrieve_context with no driver and empty communities returns empty context."""
    retriever = GraphRetriever(driver=None, communities=[])
    context = retriever.retrieve_context(["anything"])
    assert len(context["entities"]) == 0
    assert len(context["subgraph_nodes"]) == 0


def test_retriever_no_driver_graceful():
    """Retriever without driver should return empty results without crashing."""
    retriever = GraphRetriever(driver=None, communities=[])
    assert retriever.entity_match(["test"]) == []
    # Without driver, neighbor_expansion returns seed nodes as fallback
    assert retriever.neighbor_expansion(["test"]) == {"test"}
    assert retriever.multi_hop_path(["test"]) == []
