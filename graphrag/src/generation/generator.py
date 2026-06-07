import logging
from typing import Any, Optional

from graphrag.src.config.settings import LLMConfig, Settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是一个知识渊博的助手，需要使用提供的上下文来回答问题。

上下文包括：
1. **文本块** — 相关的文档摘要
2. **知识图谱** — 从文档中提取的实体及其关系
3. **社区摘要** — 相关实体的分组

回答时：
- 主要基于检索到的上下文来回答
- 如果上下文信息不足，请如实说明
- 引用具体信息时请注明来源块 ID
- 回答要简洁准确

检索到的上下文：
{context}

请基于以上上下文回答以下问题。"""

_COT_SYSTEM_PROMPT = """你是一个知识渊博的助手，需要使用提供的上下文来回答问题。

上下文包括：
1. **文本块** — 相关的文档摘要
2. **知识图谱** — 从文档中提取的实体及其关系
3. **社区摘要** — 相关实体的分组

请按以下步骤回答：

**第一步：分析问题**
- 明确问题的核心诉求
- 确定需要查找的关键信息

**第二步：检索上下文**
- 从文本块中找到相关段落
- 从知识图谱中找到相关实体和关系
- 从社区摘要中找到相关分组

**第三步：推理分析**
- 综合上下文信息进行推理
- 如果有多条相关信息，进行对比和验证
- 如果上下文信息不足，如实说明

**第四步：给出答案**
- 基于推理结果给出最终答案
- 引用具体信息时请注明来源块 ID
- 回答要简洁准确

检索到的上下文：
{context}

请基于以上上下文，按照四个步骤回答以下问题。
最终答案请以「**最终答案：**」开头。"""


class Generator:
    """使用 LLM 结合检索到的图谱和文本上下文生成答案。"""

    def __init__(self, config: LLMConfig | None = None, client: Any = None):
        self.config = config or LLMConfig()
        self.client = client

    def _build_context(self, graph_context: dict, text_chunks: list[dict]) -> str:
        """从图谱检索结果和文本块构建格式化的上下文字符串。"""
        parts: list[str] = []

        # 文本块
        if text_chunks:
            parts.append("=== Text Chunks ===")
            for i, chunk in enumerate(text_chunks):
                chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                text = chunk.get("text", "")
                parts.append(f"[{chunk_id}]: {text}")
            parts.append("")

        # 图谱实体
        entities = graph_context.get("entities", [])
        if entities:
            parts.append("=== Knowledge Graph Entities ===")
            for entity in entities:
                parts.append(f"- {entity}")
            parts.append("")

        # 图谱边
        edges = graph_context.get("subgraph_edges", [])
        if edges:
            parts.append("=== Entity Relations ===")
            for edge in edges[:30]:  # limit to 30 edges
                parts.append(
                    f"- {edge['source']} --[{edge.get('relation', 'related')}]--> {edge['target']}"
                )
            parts.append("")

        # 实体间路径
        paths = graph_context.get("paths", [])
        if paths:
            parts.append("=== Entity Paths ===")
            for p in paths:
                parts.append(" -> ".join(p))
            parts.append("")

        # 社区
        communities = graph_context.get("communities", [])
        if communities:
            parts.append("=== Communities ===")
            for i, comm in enumerate(communities):
                parts.append(f"Community {i}: {', '.join(comm[:10])}")
            parts.append("")

        return "\n".join(parts)

    def generate(
        self,
        question: str,
        graph_context: dict,
        text_chunks: list[dict],
        chain_of_thought: bool = False,
    ) -> str:
        """使用 LLM 生成答案。

        Args:
            question: 用户问题。
            graph_context: 图谱检索结果。
            text_chunks: 相关文本块。
            chain_of_thought: 是否启用思维链（CoT），让 LLM 先推理再给出最终答案。
        """
        context_str = self._build_context(graph_context, text_chunks)

        if chain_of_thought:
            system_prompt = _COT_SYSTEM_PROMPT.format(context=context_str)
        else:
            system_prompt = _SYSTEM_PROMPT.format(context=context_str)

        user_prompt = system_prompt + f"\n\nQuestion: {question}"

        if self.client is None:
            logger.warning("没有可用的 LLM 客户端；返回上下文摘要")
            return self._fallback_response(graph_context, text_chunks)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            answer = response.choices[0].message.content
            logger.info("生成的答案（%d 字符）", len(answer))
            return answer or ""
        except Exception as e:
            logger.error("LLM 生成失败: %s", e)
            return self._fallback_response(graph_context, text_chunks)

    def _fallback_response(self, graph_context: dict, text_chunks: list[dict]) -> str:
        """回退方案：无 LLM 时汇总上下文。"""
        entities = graph_context.get("entities", [])
        chunks_count = len(text_chunks)
        if entities:
            return (
                f"找到 {len(entities)} 个相关实体和 {chunks_count} 个文本块。"
                f"实体: {', '.join(entities[:5])}。"
                "请配置 LLM 提供者以生成完整答案。"
            )
        return (
            "未找到相关上下文。请提供文档或配置 LLM 提供者。"
        )
