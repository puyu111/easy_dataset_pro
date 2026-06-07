import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

from graphrag.src.chunking.semantic_chunker import TextChunk
from graphrag.src.config.settings import GraphConfig

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str = ""
    description: str = ""
    chunk_refs: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    source: str
    target: str
    relation: str = ""
    description: str = ""
    chunk_refs: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphDocument:
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


_ENTITY_PATTERN = re.compile(
    r"\((\w+)\s*-\s*(\w+)\s*-\s*(.+?)\)", re.DOTALL
)
_RELATION_PATTERN = re.compile(
    r"\[(\w+)\s*->\s*(\w+)\s*:\s*(.+?)\]", re.DOTALL
)

# Cypher queries used by GraphBuilder
_MERGE_NODE = """
MERGE (n:Entity {name: $name})
SET n.type = COALESCE($type, n.type),
    n.description = COALESCE($description, n.description),
    n.chunk_refs = CASE
        WHEN $chunk_id IS NOT NULL THEN
            CASE WHEN n.chunk_refs IS NULL THEN [$chunk_id]
                 ELSE n.chunk_refs + $chunk_id
            END
        ELSE n.chunk_refs
    END
"""

_MERGE_RELATION = """
MERGE (s:Entity {name: $source})
MERGE (t:Entity {name: $target})
MERGE (s)-[r:RELATED]->(t)
SET r.relation = COALESCE($relation, r.relation),
    r.description = COALESCE($description, r.description),
    r.chunk_refs = CASE
        WHEN $chunk_id IS NOT NULL THEN
            CASE WHEN r.chunk_refs IS NULL THEN [$chunk_id]
                 ELSE r.chunk_refs + $chunk_id
            END
        ELSE r.chunk_refs
    END
"""

_SET_COMMUNITY = """
MATCH (n:Entity {name: $name})
SET n.community_id = $community_id
"""

_CLEAR_COMMUNITIES = """
MATCH (n:Entity)
REMOVE n.community_id
"""

_GET_ALL_NODES = """
MATCH (n:Entity)
RETURN n.name AS name, n.type AS type, n.description AS description,
       n.community_id AS community_id
"""

_GET_ALL_RELATIONS = """
MATCH (s:Entity)-[r:RELATED]->(t:Entity)
RETURN s.name AS source, t.name AS target, r.relation AS relation,
       r.description AS description
"""


class GraphBuilder:
    """使用 LLM 从文本块中提取信息构建知识图谱（Neo4j 持久化）。"""

    def __init__(
        self,
        config: GraphConfig | None = None,
        llm_client: Any = None,
        driver: Any = None,
    ):
        self.config = config or GraphConfig()
        self.llm_client = llm_client
        self.driver = driver
        self.communities: list[set[str]] = []

    def build(self, chunks: list[TextChunk], resume: bool = False) -> None:
        """从文本块列表构建知识图谱。

        Args:
            chunks: 文本块列表。
            resume: 是否断点续传，跳过已在 Neo4j 中的 chunk。
        """
        if self.driver is None:
            logger.warning("Neo4j driver 不可用，无法构建图谱")
            return

        if resume:
            done_ids = self._get_processed_chunk_ids()
            logger.info("断点续传：已有 %d 个块的实体数据", len(done_ids))
        else:
            self._clear_old_data()
            done_ids = set()

        total = len(chunks)
        skipped = 0
        for i, chunk in enumerate(chunks, 1):
            if chunk.chunk_id in done_ids:
                skipped += 1
                continue

            logger.info("[%d/%d] 提取实体: %s", i, total, chunk.source)
            doc = self._extract_from_chunk(chunk)
            self._add_to_graph(doc, chunk.chunk_id)

            # 打印提取结果
            for ent in doc.entities:
                logger.info("  实体: %s (%s) - %s", ent.name, ent.type, ent.description[:60] if ent.description else "")
            for rel in doc.relations:
                logger.info("  关系: %s --[%s]--> %s", rel.source, rel.relation, rel.target)
            logger.info("  -> %d 实体, %d 关系", len(doc.entities), len(doc.relations))

        if skipped:
            logger.info("跳过 %d 个已有实体的块", skipped)

        logger.info("实体提取完成，开始社区检测...")
        self._detect_communities()

        node_count = self._count_nodes()
        edge_count = self._count_edges()
        logger.info(
            "图谱构建完成: %d 个节点, %d 条边, %d 个社区",
            node_count,
            edge_count,
            len(self.communities),
        )

    def _clear_old_data(self) -> None:
        """清理 Neo4j 中的旧数据。"""
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run("MATCH (n:Entity) DETACH DELETE n")
        logger.info("已清理 Neo4j 中的旧数据")

    def _get_processed_chunk_ids(self) -> set[str]:
        """查询 Neo4j 中已有实体数据的 chunk_id 集合。"""
        if self.driver is None:
            return set()
        ids: set[str] = set()
        with self.driver.session() as session:
            result = session.run(
                "MATCH (n:Entity) WHERE n.chunk_refs IS NOT NULL "
                "UNWIND n.chunk_refs AS cid RETURN DISTINCT cid"
            )
            for record in result:
                ids.add(record["cid"])
        return ids

    def _count_nodes(self) -> int:
        if self.driver is None:
            return 0
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
            return result.single()["cnt"]

    def _count_edges(self) -> int:
        if self.driver is None:
            return 0
        with self.driver.session() as session:
            result = session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS cnt")
            return result.single()["cnt"]

    def _extract_from_chunk(self, chunk: TextChunk, retries: int = 3) -> GraphDocument:
        """使用 LLM 从文本块中提取实体和关系。"""
        import time

        if self.llm_client is None:
            return self._rule_based_extraction(chunk)

        # 截断超长文本，避免 LLM 超时
        text = chunk.text[:4000] if len(chunk.text) > 4000 else chunk.text

        prompt = (
            "从以下文本中提取实体及其关系。"
            "返回一个包含两个列表的 JSON 对象：\n"
            '- "entities": list of {"name": str, "type": str, "description": str}\n'
            '- "relations": list of {"source": str, "target": str, "relation": str, "description": str}\n\n'
            f"文本:\n{text}"
        )

        for attempt in range(retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.config.extraction_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个知识图谱提取助手。"
                            "请以 JSON 格式提取实体和关系。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=60,
                )
                content = response.choices[0].message.content
                data = json.loads(content)
                entities = [
                    Entity(
                        name=e["name"],
                        type=e.get("type", ""),
                        description=e.get("description", ""),
                        chunk_refs=[chunk.chunk_id],
                    )
                    for e in data.get("entities", [])
                ]
                relations = [
                    Relation(
                        source=r["source"],
                        target=r["target"],
                        relation=r.get("relation", ""),
                        description=r.get("description", ""),
                        chunk_refs=[chunk.chunk_id],
                    )
                    for r in data.get("relations", [])
                ]
                return GraphDocument(entities=entities, relations=relations)
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning("  重试 %d/%d (%ds): %s", attempt + 1, retries, wait, e)
                    time.sleep(wait)
                else:
                    logger.warning("块 %s 的 LLM 提取失败，回退到规则提取: %s", chunk.chunk_id, e)
                    return self._rule_based_extraction(chunk)

        return self._rule_based_extraction(chunk)

    def _rule_based_extraction(self, chunk: TextChunk) -> GraphDocument:
        """回退方案：使用正则模式进行简单的名词短语提取。"""
        text = chunk.text
        entities: list[Entity] = []
        relations: list[Relation] = []

        for match in _ENTITY_PATTERN.finditer(text):
            name = match.group(1)
            ent_type = match.group(2)
            desc = match.group(3).strip()
            entities.append(
                Entity(
                    name=name,
                    type=ent_type,
                    description=desc,
                    chunk_refs=[chunk.chunk_id],
                )
            )

        for match in _RELATION_PATTERN.finditer(text):
            source = match.group(1)
            target = match.group(2)
            rel = match.group(3).strip()
            relations.append(
                Relation(
                    source=source,
                    target=target,
                    relation=rel,
                    chunk_refs=[chunk.chunk_id],
                )
            )

        return GraphDocument(entities=entities, relations=relations)

    def _add_to_graph(self, doc: GraphDocument, chunk_id: str) -> None:
        """将提取的实体和关系通过 Cypher MERGE 写入 Neo4j。"""
        if self.driver is None:
            logger.warning("Neo4j driver 不可用，跳过图写入")
            return

        with self.driver.session() as session:
            for entity in doc.entities:
                session.run(
                    _MERGE_NODE,
                    name=entity.name,
                    type=entity.type or None,
                    description=entity.description or None,
                    chunk_id=chunk_id,
                )

            for relation in doc.relations:
                session.run(
                    _MERGE_RELATION,
                    source=relation.source,
                    target=relation.target,
                    relation=relation.relation or None,
                    description=relation.description or None,
                    chunk_id=chunk_id,
                )

    def _detect_communities(self) -> None:
        """从 Neo4j 拉取子图到 NetworkX，运行社区检测，写回 community_id。"""
        if self.driver is None:
            self.communities = []
            return

        # 从 Neo4j 拉取全图到 NetworkX
        G = nx.Graph()
        with self.driver.session() as session:
            for record in session.run(_GET_ALL_NODES):
                G.add_node(record["name"])

            for record in session.run(_GET_ALL_RELATIONS):
                G.add_edge(record["source"], record["target"])

        if G.number_of_nodes() < 2:
            self.communities = [{n} for n in G.nodes]
            self._write_communities()
            return

        try:
            from networkx.algorithms.community import greedy_modularity_communities

            communities = list(
                greedy_modularity_communities(
                    G,
                    resolution=self.config.community_detection.resolution,
                )
            )
            self.communities = [set(c) for c in communities]
        except Exception as e:
            logger.warning("社区检测失败: %s", e)
            self.communities = [{n} for n in G.nodes]

        self._write_communities()

    def _write_communities(self) -> None:
        """将 community_id 写回 Neo4j。"""
        if self.driver is None:
            return
        with self.driver.session() as session:
            session.run(_CLEAR_COMMUNITIES)
            for idx, comm in enumerate(self.communities):
                for node_name in comm:
                    session.run(_SET_COMMUNITY, name=node_name, community_id=idx)

    def get_node_community(self, node: str) -> Optional[int]:
        """返回给定节点所属的社区索引，如果不存在则返回 None。"""
        for idx, comm in enumerate(self.communities):
            if node in comm:
                return idx
        return None

    def export_gexf(self, path: str) -> None:
        """从 Neo4j 拉全图到 NetworkX 再导出为 GEXF 格式（用于 Gephi）。"""
        G = nx.Graph()
        if self.driver:
            with self.driver.session() as session:
                for record in session.run(_GET_ALL_NODES):
                    attrs = {"type": record["type"] or "", "description": record["description"] or ""}
                    if record["community_id"] is not None:
                        attrs["community_id"] = record["community_id"]
                    G.add_node(record["name"], **attrs)

                for record in session.run(_GET_ALL_RELATIONS):
                    G.add_edge(
                        record["source"],
                        record["target"],
                        relation=record["relation"] or "",
                        description=record["description"] or "",
                    )
        else:
            logger.warning("Neo4j driver 不可用，导出空图")

        nx.write_gexf(G, path)
        logger.info("图已导出到 %s（%d 个节点，%d 条边）", path, G.number_of_nodes(), G.number_of_edges())
