#!/usr/bin/env python3
"""批量问答脚本：读取 dataset_all.jsonl，通过 GraphRAG 回答每个问题，将答案写入数据集。

用法:
    # 先构建 RAG（只需一次）
    python run.py rag-build

    # 批量问答（加载已构建的状态）
    python batch_query.py                           # 处理全部问题
    python batch_query.py --limit 100               # 只处理前 100 条
    python batch_query.py --resume                  # 断点续传（跳过已有答案的行）
    python batch_query.py --output output.jsonl     # 指定输出文件
    python batch_query.py --state output/rag_state.pkl  # 指定 RAG 状态文件
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from graphrag.src.logging_setup import setup_logging
from graphrag.src.main import GraphRAG
from graphrag.src.config.settings import load_config

logger = logging.getLogger("batch_query")


def main():
    parser = argparse.ArgumentParser(description="批量问答：用 GraphRAG 回答数据集问题")
    parser.add_argument("--config", default="graphrag/config/config.yaml", help="配置文件路径")
    parser.add_argument("--state", default=None, help="RAG 状态文件路径（默认从配置读取）")
    parser.add_argument("--dataset", default=None, help="数据集路径（默认从配置读取）")
    parser.add_argument("--output", default=None, help="输出文件路径（默认从配置读取）")
    parser.add_argument("--limit", type=int, default=0, help="处理条数限制（0=全部）")
    parser.add_argument("--resume", action="store_true", help="断点续传（跳过已有 answer 的行）")
    parser.add_argument("--offset", type=int, default=0, help="跳过前 N 条")
    parser.add_argument("--rebuild", action="store_true", help="强制重新构建 RAG（忽略已有状态文件）")
    parser.add_argument("--chain-of-thought", action="store_true", default=None, help="启用思维链（CoT），覆盖配置文件设置")
    parser.add_argument("--no-chain-of-thought", action="store_true", help="禁用思维链，覆盖配置文件设置")
    args = parser.parse_args()

    # 加载配置并初始化日志
    config_path = args.config
    settings = load_config(config_path)
    setup_logging(settings.logging)

    # 命令行参数优先，否则从配置读取
    batch_cfg = settings.batch
    if args.dataset is None:
        args.dataset = batch_cfg.dataset
    if args.output is None:
        args.output = batch_cfg.output
    if args.state is None:
        args.state = batch_cfg.state
    if not args.resume:
        args.resume = batch_cfg.resume
    save_interval = batch_cfg.save_interval
    chain_of_thought = batch_cfg.chain_of_thought
    # 命令行参数覆盖配置
    if args.chain_of_thought:
        chain_of_thought = True
    elif args.no_chain_of_thought:
        chain_of_thought = False

    # 读取完整数据集（始终从原始数据集加载）
    dataset_path = Path(args.dataset)
    output_path = Path(args.output) if args.output else dataset_path

    if not dataset_path.exists():
        logger.error("数据集文件不存在: %s", dataset_path)
        sys.exit(1)

    all_records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))
    logger.info("数据集共 %d 条记录", len(all_records))

    # 断点续传：从输出文件合并已有答案
    if args.resume and output_path.exists():
        existing = {}
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    q = r.get("question", "")
                    ans = r.get("answer", "")
                    # 跳过 fallback 答案
                    if ans and not (ans.startswith("找到 ") and "请配置 LLM" in ans):
                        existing[q] = ans
        merged = 0
        for r in all_records:
            q = r.get("question", "")
            if q in existing:
                r["answer"] = existing[q]
                merged += 1
        logger.info("断点续传：合并 %d 条已有答案", merged)

    # 确定需要处理的记录（应用 offset/limit/跳过已有答案）
    process_indices = list(range(len(all_records)))
    if args.offset > 0:
        process_indices = [i for i in process_indices if i >= args.offset]
    if args.limit > 0:
        process_indices = process_indices[:args.limit]
    if args.resume:
        process_indices = [i for i in process_indices if not all_records[i].get("answer")]

    logger.info("需要处理 %d 条记录", len(process_indices))

    if not process_indices:
        logger.info("没有需要处理的记录")
        return

    # 初始化 RAG：优先加载已构建的状态，否则从头构建
    state_path = Path(args.state)
    if state_path.exists() and not args.rebuild:
        logger.info("加载已构建的 RAG 状态: %s", state_path)
        rag = GraphRAG.load(str(state_path), config_path=config_path)
    else:
        logger.info("正在初始化 GraphRAG...")
        rag = GraphRAG(config_path=config_path)

        # 加载文档
        docs = {}
        doc_dir = Path(settings.documents.directory)
        if doc_dir.is_dir():
            for fpath in sorted(doc_dir.iterdir()):
                if fpath.is_file():
                    try:
                        text = fpath.read_text(encoding="utf-8")
                        docs[fpath.stem] = text
                    except Exception:
                        try:
                            text = fpath.read_text(encoding="gbk")
                            docs[fpath.stem] = text
                        except Exception:
                            logger.warning("无法读取文件: %s", fpath)

        if not docs:
            logger.error("未找到可加载的文档")
            sys.exit(1)

        logger.info("已加载 %d 篇文档", len(docs))
        rag.load_documents(docs)

        logger.info("正在切分文档...")
        chunks = rag.chunk_documents()
        logger.info("共 %d 个文本块", len(chunks))

        logger.info("正在构建知识图谱...")
        rag.build_graph()

        # 保存状态供下次使用
        state_path.parent.mkdir(parents=True, exist_ok=True)
        rag.save(str(state_path))
        logger.info("RAG 状态已保存到: %s", state_path)

    stats = rag.get_statistics()
    logger.info("图谱就绪: %s", stats)

    # 批量问答
    total = len(process_indices)
    answered = 0
    failed = 0
    start_time = time.time()

    logger.info("")
    logger.info("=" * 60)
    logger.info("开始批量问答 (%d 条)", total)
    logger.info("思维链: %s", "启用" if chain_of_thought else "关闭")
    logger.info("=" * 60)

    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 20

    for processed, idx in enumerate(process_indices):
        record = all_records[idx]
        question = record.get("question", "")
        if not question:
            continue

        try:
            answer = rag.query(question, chain_of_thought=chain_of_thought)
            if answer.startswith("找到 ") and "请配置 LLM" in answer:
                logger.warning("[%d/%d] LLM 未返回有效答案，标记为空", processed + 1, total)
                record["answer"] = ""
                failed += 1
                consecutive_errors += 1
            else:
                record["answer"] = answer
                answered += 1
                consecutive_errors = 0
                logger.info("[%d/%d] Q: %s", processed + 1, total, question[:60])
                logger.info("[%d/%d] A: %s", processed + 1, total, answer[:200])
        except Exception as e:
            logger.error("[%d/%d] 失败: %s -> %s", processed + 1, total, question[:50], e)
            record["answer"] = ""
            failed += 1
            consecutive_errors += 1

        # 连续失败过多则停止
        if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
            logger.error("连续 %d 次失败，停止运行。请检查 LLM API 余额或配置。", consecutive_errors)
            _save_results(all_records, output_path)
            sys.exit(1)

        if (processed + 1) % 10 == 0 or (processed + 1) == total:
            elapsed = time.time() - start_time
            avg = elapsed / (processed + 1)
            eta = avg * (total - processed - 1)
            logger.info(
                "[%d/%d] 已回答=%d 失败=%d 耗时=%.0fs ETA=%.0fs",
                processed + 1, total, answered, failed, elapsed, eta,
            )

        # 定期保存完整数据集
        if (processed + 1) % save_interval == 0:
            _save_results(all_records, output_path)

    # 最终保存
    _save_results(all_records, output_path)

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("批量问答完成")
    logger.info("=" * 60)
    logger.info("总条数:   %d", total)
    logger.info("成功:     %d", answered)
    logger.info("失败:     %d", failed)
    logger.info("总耗时:   %.1fs", elapsed)
    logger.info("平均每条: %.2fs", elapsed / total if total else 0)
    logger.info("输出文件: %s", output_path)
    logger.info("=" * 60)


def _save_results(records: list[dict], output_path: Path):
    """保存结果到 JSONL 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
