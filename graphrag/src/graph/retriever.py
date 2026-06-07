import logging
from typing import Any, Optional

from graphrag.src.config.settings import RetrievalConfig

logger = logging.getLogger(__name__)


class GraphRetriever:
    """知识图谱上的多策略检索器（Neo4j Cypher 查询）。"""

    def __init__(
        self,
        driver: Any,
        communities: list[set[str]],
        config: RetrievalConfig | None = None,
    ):
        self.driver = driver
        self.communities = communities
        self.config = config or RetrievalConfig()

    def entity_match(self, query_terms: list[str]) -> list[str]:
        """查找名称与查询实体匹配的节点（不区分大小写）。"""
        if self.driver is None:
            return []
        matched: list[str] = []
        with self.driver.session() as session:
            for term in query_terms:
                result = session.run(
                    "MATCH (n:Entity) WHERE toLower(n.name) CONTAINS toLower($term) "
                    "RETURN n.name AS name LIMIT $limit",
                    term=term,
                    limit=self.config.max_entities,
                )
                matched.extend(record["name"] for record in result)
        # 去重并保持顺序
        seen: set[str] = set()
        unique: list[str] = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique[: self.config.max_entities]

    def neighbor_expansion(
        self, seed_nodes: list[str], hops: Optional[int] = None
    ) -> set[str]:
        """从种子节点向外扩展 N 跳。"""
        if self.driver is None:
            return set(seed_nodes)
        if hops is None:
            hops = self.config.neighbor_hops

        expanded: set[str] = set()
        with self.driver.session() as session:
            for seed in seed_nodes:
                result = session.run(
                    f"MATCH (n:Entity {{name: $seed}})-[*0..{hops}]-(m:Entity) "
                    "RETURN DISTINCT m.name AS name",
                    seed=seed,
                )
                expanded.update(record["name"] for record in result)
        return expanded

    def community_retrieval(self, seed_nodes: list[str]) -> list[set[str]]:
        """查找包含任意种子节点的社区。"""
        seed_set = set(seed_nodes)
        matched: list[set[str]] = []
        for comm in self.communities:
            if comm & seed_set:
                matched.append(comm)
                if len(matched) >= self.config.max_community_results:
                    break
        return matched

    def multi_hop_path(self, entities: list[str]) -> list[list[str]]:
        """查找每对查询实体之间的最短路径。"""
        if self.driver is None:
            return []
        paths: list[list[str]] = []
        with self.driver.session() as session:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    result = session.run(
                        "MATCH p = shortestPath("
                        "(s:Entity {name: $source})-[*..50]-(t:Entity {name: $target})"
                        ") "
                        "RETURN [node IN nodes(p) | node.name] AS path_names",
                        source=entities[i],
                        target=entities[j],
                    )
                    record = result.single()
                    if record:
                        paths.append(record["path_names"])
        return paths

    def retrieve_context(
        self, query_terms: list[str]
    ) -> dict:
        """运行所有检索策略并合并结果。

        返回:
            dict，包含以下键：entities, subgraph_nodes, subgraph_edges, communities, paths
        """
        entities = self.entity_match(query_terms)
        logger.info("实体匹配找到 %d 个节点", len(entities))

        if not entities:
            return {
                "entities": [],
                "subgraph_nodes": [],
                "subgraph_edges": [],
                "communities": [],
                "paths": [],
            }

        subgraph_nodes = self.neighbor_expansion(entities)
        logger.info("邻居扩展: %d 个节点", len(subgraph_nodes))

        # 从 Neo4j 获取子图边信息
        subgraph_edges: list[dict] = []
        if self.driver is not None and subgraph_nodes:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (s:Entity)-[r:RELATED]->(t:Entity) "
                    "WHERE s.name IN $nodes AND t.name IN $nodes "
                    "RETURN s.name AS source, t.name AS target, "
                    "       r.relation AS relation",
                    nodes=list(subgraph_nodes),
                )
                subgraph_edges = [
                    {
                        "source": record["source"],
                        "target": record["target"],
                        "relation": record["relation"] or "",
                    }
                    for record in result
                ]

        communities = self.community_retrieval(entities)
        logger.info("社区检索: %d 个社区", len(communities))

        paths = self.multi_hop_path(entities)
        logger.info("多跳路径: 找到 %d 条路径", len(paths))

        return {
            "entities": entities,
            "subgraph_nodes": list(subgraph_nodes),
            "subgraph_edges": subgraph_edges,
            "communities": [list(c) for c in communities],
            "paths": paths,
        }
