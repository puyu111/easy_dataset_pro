#!/usr/bin/env python3
"""GraphRAG 启动脚本 - 支持文档预处理、命令行问答和交互式会话。"""

import argparse
import logging
import sys
from pathlib import Path

from graphrag.src.logging_setup import setup_logging
from graphrag.src.main import GraphRAG
from graphrag.src.preprocessing.pipeline import PreprocessingPipeline
from graphrag.src.preprocessing.dataset_generator import DatasetGenerator

logger = logging.getLogger("run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GraphRAG - 基于知识图谱的检索增强生成系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ========== 预处理子命令 ==========
    preproc = subparsers.add_parser("preprocess", help="文档预处理流程")
    preproc.add_argument("--docs-dir", default=None,
                         help="原始文档目录")
    preproc.add_argument("--md-dir", default=None,
                         help="Markdown 输出目录")
    preproc.add_argument("--filtered-dir", default=None,
                         help="过滤后 Markdown 输出目录")
    preproc.add_argument("--compressed-dir", default=None,
                         help="压缩后 Markdown 输出目录")
    preproc.add_argument("--output-base", default=None,
                         help="输出根目录")
    preproc.add_argument("--datatrove-dir", default=None,
                         help="datatrove 源码目录")
    preproc.add_argument("--stage", choices=["convert", "quality", "compress", "all"],
                         default="all", help="要运行的阶段 (默认: all)")
    preproc.add_argument("--no-ocr", action="store_true",
                         help="关闭 OCR")
    preproc.add_argument("--force-ocr", action="store_true",
                         help="强制 OCR")
    preproc.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                         help="推理设备")
    preproc.add_argument("--quality-mode", choices=["stats", "filter", "full"],
                         default="full", help="质量过滤模式")
    preproc.add_argument("--min-chars", type=int, default=200,
                         help="最小字符数阈值")
    preproc.add_argument("--min-quality", type=float, default=0.4,
                         help="最低质量分数")
    preproc.add_argument("--max-chunk-tokens", type=int, default=24000,
                         help="LLM 压缩每段最大 token 数")
    preproc.add_argument("--concurrency", type=int, default=1,
                         help="LLM 压缩并发数")
    preproc.add_argument("--dry-run", action="store_true",
                         help="LLM 压缩仅预估不调用")
    preproc.add_argument("--no-chunk", action="store_true",
                         help="LLM 压缩不分块，整篇文档一次发送")
    preproc.add_argument("--no-resume", action="store_true",
                         help="禁用断点续传")
    preproc.add_argument("--config",
                         help="配置文件路径（用于读取预处理配置）")

    # ========== 数据集生成子命令 ==========
    dataset = subparsers.add_parser("dataset", help="从 Markdown 文档生成数据集问答对")
    dataset.add_argument("--input-dir", default=None,
                         help="输入目录（默认: output/llm_compressed）")
    dataset.add_argument("--output-dir", default=None,
                         help="输出目录（默认: output/dataset）")
    dataset.add_argument("--files", type=str, default=None,
                         help="文件范围，如 '1-5', '3,7,9'")
    dataset.add_argument("--questions-per-chunk", type=int, default=None,
                         help="每段生成的问题数")
    dataset.add_argument("--max-chunk-tokens", type=int, default=None,
                         help="每段最大 token 数")
    dataset.add_argument("--concurrency", type=int, default=None,
                         help="并发数")
    dataset.add_argument("--dry-run", action="store_true",
                         help="仅预估不调用 LLM")
    dataset.add_argument("--no-resume", action="store_true",
                         help="禁用断点续传")
    dataset.add_argument("--no-chunk", action="store_true",
                         help="不分块，整篇文档一次发送")
    dataset.add_argument("--config",
                         help="配置文件路径")

    # ========== RAG 构建子命令 ==========
    rag_build = subparsers.add_parser("rag-build", help="构建 RAG（加载文档、切分、建图谱）并保存状态")
    rag_build.add_argument(
        "-c", "--config",
        default="graphrag/config/config.yaml",
        help="配置文件路径（默认: graphrag/config/config.yaml）",
    )
    rag_build.add_argument(
        "-f", "--file",
        action="append", dest="files",
        help="要加载的文档文件路径（可多次使用）",
    )
    rag_build.add_argument(
        "-d", "--document",
        action="append", dest="documents",
        nargs=2, metavar=("NAME", "TEXT"),
        help="直接传入文档内容，格式: 名称 文本（可多次使用）",
    )
    rag_build.add_argument(
        "-o", "--output",
        default="output/rag_state.pkl",
        help="RAG 状态保存路径（默认: output/rag_state.pkl）",
    )
    rag_build.add_argument(
        "--export", metavar="PATH",
        help="完成后将知识图谱导出为 GEXF 文件",
    )
    rag_build.add_argument(
        "--stats", action="store_true",
        help="显示统计信息",
    )

    # ========== RAG 查询子命令 ==========
    rag_query = subparsers.add_parser("rag-query", help="加载已构建的 RAG 状态并进行问答")
    rag_query.add_argument(
        "-c", "--config",
        default="graphrag/config/config.yaml",
        help="配置文件路径（默认: graphrag/config/config.yaml）",
    )
    rag_query.add_argument(
        "-s", "--state",
        default="output/rag_state.pkl",
        help="RAG 状态文件路径（默认: output/rag_state.pkl）",
    )
    rag_query.add_argument(
        "-q", "--query",
        help="单次查询模式：直接提问并退出",
    )
    rag_query.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互式问答模式",
    )
    rag_query.add_argument(
        "--chain-of-thought",
        action="store_true",
        default=None,
        help="启用思维链（CoT），覆盖配置文件设置",
    )
    rag_query.add_argument(
        "--no-chain-of-thought",
        action="store_true",
        help="禁用思维链，覆盖配置文件设置",
    )

    # ========== RAG 旧版子命令（兼容） ==========
    rag = subparsers.add_parser("rag", help="运行 GraphRAG 问答系统（兼容旧版，构建+查询一步完成）")
    rag.add_argument(
        "-c", "--config",
        default="graphrag/config/config.yaml",
        help="配置文件路径（默认: graphrag/config/config.yaml）",
    )
    rag.add_argument(
        "-f", "--file",
        action="append", dest="files",
        help="要加载的文档文件路径（可多次使用）",
    )
    rag.add_argument(
        "-d", "--document",
        action="append", dest="documents",
        nargs=2, metavar=("NAME", "TEXT"),
        help="直接传入文档内容，格式: 名称 文本（可多次使用）",
    )
    rag.add_argument(
        "-q", "--query",
        help="单次查询模式：直接提问并退出",
    )
    rag.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互式问答模式",
    )
    rag.add_argument(
        "--export", metavar="PATH",
        help="完成后将知识图谱导出为 GEXF 文件",
    )
    rag.add_argument(
        "--stats", action="store_true",
        help="显示统计信息",
    )
    rag.add_argument(
        "--chain-of-thought",
        action="store_true",
        default=None,
        help="启用思维链（CoT），覆盖配置文件设置",
    )
    rag.add_argument(
        "--no-chain-of-thought",
        action="store_true",
        help="禁用思维链，覆盖配置文件设置",
    )

    return parser


def demo_documents() -> dict[str, str]:
    """当用户未提供任何文档时返回预置示例文档。"""
    return {
        "demo_ai": (
            "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统。"
            "机器学习是AI的核心子领域，通过数据训练模型来做出决策。"
            "深度学习是机器学习的子集，使用多层神经网络处理复杂模式。"
            "自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。"
            "Transformer架构是现代NLP的基础，BERT和GPT是其代表性模型。"
        ),
        "demo_company": (
            "深度求索（DeepSeek）是一家专注于人工智能研究和应用的中国科技公司。"
            "公司开发了大语言模型DeepSeek系列，包括DeepSeek-V2和DeepSeek-V3等版本。"
            "DeepSeek模型在多项基准测试中表现出色，支持文本理解和代码生成等任务。"
            "深度求索致力于推动AI技术的创新和应用，为用户提供高质量的AI服务。"
        ),
    }


def run_preprocessing(args) -> None:
    """运行预处理子命令。"""
    # 如果指定了配置文件，从配置中读取预处理设置
    config_path = args.config or "graphrag/config/config.yaml"
    settings = None
    if Path(config_path).exists():
        from graphrag.src.config.settings import load_config
        settings = load_config(config_path)
        setup_logging(settings.logging)
        preproc_cfg = settings.preprocessing
    else:
        from graphrag.src.config.settings import LoggingConfig
        setup_logging(LoggingConfig())
        preproc_cfg = type("cfg", (), {
            "docs_dir": " ", "md_dir": "", "filtered_dir": "",
            "compressed_dir": "", "output_base": "", "datatrove_dir": "",
            "convert_do_ocr": True, "convert_force_ocr": False,
            "convert_max_pages": 20, "convert_device": "auto", "convert_resume": True,
            "quality_mode": "full", "quality_min_chars": 200, "quality_min_score": 0.4,
            "compress_max_chunk_tokens": 24000, "compress_max_output_tokens": 8192,
            "compress_concurrency": 1, "convert_formats": None,
        })()

    pipeline = PreprocessingPipeline(
        docs_dir=args.docs_dir or preproc_cfg.docs_dir,
        md_dir=args.md_dir or preproc_cfg.md_dir or None,
        filtered_dir=args.filtered_dir or preproc_cfg.filtered_dir or None,
        compressed_dir=args.compressed_dir or preproc_cfg.compressed_dir or None,
        output_base=args.output_base or preproc_cfg.output_base or None,
        datatrove_dir=args.datatrove_dir or preproc_cfg.datatrove_dir or None,
    )

    stage = args.stage

    convert_kwargs = {
        "do_ocr": not args.no_ocr if args.no_ocr else preproc_cfg.convert_do_ocr,
        "force_ocr": args.force_ocr if args.force_ocr else preproc_cfg.convert_force_ocr,
        "max_pages": preproc_cfg.convert_max_pages,
        "device": args.device if args.device != "auto" else preproc_cfg.convert_device,
        "resume": not args.no_resume if args.no_resume else preproc_cfg.convert_resume,
    }

    quality_kwargs = {
        "mode": args.quality_mode if args.quality_mode != "full" else preproc_cfg.quality_mode,
        "min_chars": args.min_chars if args.min_chars != 200 else preproc_cfg.quality_min_chars,
        "min_quality": args.min_quality if args.min_quality != 0.4 else preproc_cfg.quality_min_score,
    }

    compress_kwargs = {
        "max_chunk_tokens": preproc_cfg.compress_max_chunk_tokens,
        "max_output_tokens": preproc_cfg.compress_max_output_tokens,
        "concurrency": args.concurrency if args.concurrency != 1 else preproc_cfg.compress_concurrency,
        "resume": not args.no_resume,
        "dry_run": args.dry_run,
        "no_chunk": args.no_chunk or preproc_cfg.compress_no_chunk,
    }

    # 从 config.yaml 的 llm 配置中读取 api_key / base_url / model
    if settings is not None:
        compress_kwargs.setdefault("api_key", settings.llm.api_key)
        compress_kwargs.setdefault("base_url", settings.llm.base_url)
        compress_kwargs.setdefault("model", settings.llm.model)

    if stage == "convert":
        pipeline.convert(**convert_kwargs)
    elif stage == "quality":
        pipeline.quality_filter(**quality_kwargs)
    elif stage == "compress":
        pipeline.compress(**compress_kwargs)
    else:  # all
        pipeline.run_all(
            convert_kwargs=convert_kwargs,
            quality_kwargs=quality_kwargs,
            compress_kwargs=compress_kwargs,
        )


def run_dataset(args) -> None:
    """运行数据集生成子命令。"""
    config_path = args.config or "graphrag/config/config.yaml"
    settings = None
    if Path(config_path).exists():
        from graphrag.src.config.settings import load_config
        settings = load_config(config_path)
        setup_logging(settings.logging)
        preproc_cfg = settings.preprocessing
    else:
        from graphrag.src.config.settings import LoggingConfig
        setup_logging(LoggingConfig())
        preproc_cfg = None

    input_dir = args.input_dir or str(Path.cwd() / "output" / "llm_compressed")
    output_dir = args.output_dir or str(Path.cwd() / "output" / "dataset")

    questions_per_chunk = args.questions_per_chunk
    max_chunk_tokens = args.max_chunk_tokens
    concurrency = args.concurrency
    no_chunk = args.no_chunk

    if preproc_cfg is not None:
        if questions_per_chunk is None:
            questions_per_chunk = preproc_cfg.dataset_questions_per_chunk
        if max_chunk_tokens is None:
            max_chunk_tokens = preproc_cfg.dataset_max_chunk_tokens
        if concurrency is None:
            concurrency = preproc_cfg.dataset_concurrency
        if not no_chunk:
            no_chunk = preproc_cfg.dataset_no_chunk

    # 设置默认值
    if questions_per_chunk is None:
        questions_per_chunk = 5
    if max_chunk_tokens is None:
        max_chunk_tokens = 24000
    if concurrency is None:
        concurrency = 1

    generator = DatasetGenerator(
        input_dir=input_dir,
        output_dir=output_dir,
        questions_per_chunk=questions_per_chunk,
        max_chunk_tokens=max_chunk_tokens,
        api_key=settings.llm.api_key if settings else None,
        base_url=settings.llm.base_url if settings else None,
        model=settings.llm.model if settings else None,
    )

    generator.run(
        file_range=args.files,
        dry_run=args.dry_run,
        concurrency=concurrency,
        resume=not args.no_resume,
        no_chunk=no_chunk,
    )


def run_rag_build(args) -> None:
    """构建 RAG：加载文档 → 切分 → 建图谱 → 保存状态。"""
    config_path = args.config
    if not Path(config_path).exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    logger.info("正在初始化 GraphRAG...")
    rag = GraphRAG(config_path=config_path)
    setup_logging(rag.settings.logging)

    # 加载文档
    docs: dict[str, str] = {}

    if args.files:
        for filepath in args.files:
            path = Path(filepath)
            if not path.exists():
                logger.warning("文件不存在，跳过: %s", filepath)
                continue
            docs[path.stem] = path.read_text(encoding="utf-8")
            logger.info("已加载文件: %s", filepath)
    elif rag.settings.documents.directory:
        doc_dir = Path(rag.settings.documents.directory)
        if not doc_dir.is_dir():
            logger.warning("文档目录不存在: %s", doc_dir)
        else:
            for fpath in sorted(doc_dir.iterdir()):
                if fpath.is_file():
                    try:
                        docs[fpath.stem] = fpath.read_text(encoding="utf-8")
                    except UnicodeDecodeError:
                        docs[fpath.stem] = fpath.read_text(encoding="gbk", errors="ignore")
                    logger.info("已加载文件: %s", fpath)

    if args.documents:
        for name, text in args.documents:
            docs[name] = text
            logger.info("已加载文档: %s", name)

    if not docs:
        logger.error("未指定任何文档，请通过 -f 或配置文件指定文档目录")
        sys.exit(1)

    rag.load_documents(docs)

    logger.info("正在切分文档...")
    chunks = rag.chunk_documents(output_dir="output/chunks", resume=True)
    logger.info("共 %d 个文本块", len(chunks))

    logger.info("正在构建知识图谱...")
    rag.build_graph(resume=True)

    # 保存状态
    output_path = args.output
    rag.save(output_path)

    if args.export:
        rag.export_graph(args.export)
        logger.info("图谱已导出到: %s", args.export)

    if args.stats:
        stats = rag.get_statistics()
        print("\n=== GraphRAG 统计信息 ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

    rag.close()
    logger.info("构建完成！使用以下命令进行查询:")
    logger.info("  python run.py rag-query -s %s -q '你的问题'", output_path)
    logger.info("  python run.py rag-query -s %s -i", output_path)


def run_rag_query(args) -> None:
    """加载已构建的 RAG 状态并进行问答。"""
    config_path = args.config
    state_path = args.state

    if not Path(state_path).exists():
        logger.error("RAG 状态文件不存在: %s", state_path)
        logger.error("请先运行: python run.py rag-build")
        sys.exit(1)

    if not Path(config_path).exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    logger.info("正在加载 RAG 状态: %s", state_path)
    rag = GraphRAG.load(state_path, config_path=config_path)
    setup_logging(rag.settings.logging)

    # 确定思维链设置
    chain_of_thought = rag.settings.batch.chain_of_thought
    if getattr(args, 'chain_of_thought', None):
        chain_of_thought = True
    elif getattr(args, 'no_chain_of_thought', False):
        chain_of_thought = False

    if args.query:
        print(f"\n问题: {args.query}")
        print(f"答案: {rag.query(args.query, chain_of_thought=chain_of_thought)}\n")
    else:
        print(f"\nGraphRAG 已就绪！（思维链: {'启用' if chain_of_thought else '关闭'}，输入 exit 退出）\n")
        while True:
            try:
                question = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            answer = rag.query(question, chain_of_thought=chain_of_thought)
            print(answer)
            print()

    rag.close()


def run_rag(args) -> None:
    """运行 GraphRAG 问答系统。"""
    config_path = args.config
    if not Path(config_path).exists():
        logger.error("配置文件不存在: %s", config_path)
        sys.exit(1)

    logger.info("正在初始化 GraphRAG...")
    rag = GraphRAG(config_path=config_path)

    # 在 GraphRAG 加载配置后及时初始化日志系统
    setup_logging(rag.settings.logging)
    logger.info(
        "配置: embedder=%s, similarity=%s, llm=%s, rerank=%s",
        rag.settings.embeddings.provider,
        rag.settings.similarity.strategy,
        rag.settings.llm.provider,
        rag.settings.rerank.provider if rag.settings.rerank.enabled else "disabled",
    )

    # 加载文档
    docs: dict[str, str] = {}

    if args.files:
        for filepath in args.files:
            path = Path(filepath)
            if not path.exists():
                logger.warning("文件不存在，跳过: %s", filepath)
                continue
            docs[path.stem] = path.read_text(encoding="utf-8")
            logger.info("已加载文件: %s", filepath)
    elif rag.settings.documents.directory:
        doc_dir = Path(rag.settings.documents.directory)
        if not doc_dir.is_dir():
            logger.warning("文档目录不存在: %s", doc_dir)
        else:
            for fpath in sorted(doc_dir.iterdir()):
                if fpath.is_file():
                    docs[fpath.stem] = fpath.read_text(encoding="utf-8")
                    logger.info("已加载文件(配置): %s", fpath)

    if args.documents:
        for name, text in args.documents:
            docs[name] = text
            logger.info("已加载文档: %s", name)

    if not docs:
        logger.info("未指定文档，加载预置示例文档...")
        docs = demo_documents()

    rag.load_documents(docs)

    logger.info("正在切分文档...")
    chunks = rag.chunk_documents(output_dir="output/chunks", resume=True)
    logger.info("共 %d 个文本块", len(chunks))

    logger.info("正在构建知识图谱...")
    rag.build_graph(resume=True)

    if args.export:
        rag.export_graph(args.export)
        logger.info("图谱已导出到: %s", args.export)

    if args.stats:
        stats = rag.get_statistics()
        print("\n=== GraphRAG 统计信息 ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()

    # 确定思维链设置
    chain_of_thought = rag.settings.batch.chain_of_thought
    if getattr(args, 'chain_of_thought', None):
        chain_of_thought = True
    elif getattr(args, 'no_chain_of_thought', False):
        chain_of_thought = False

    if args.query:
        print(f"\n问题: {args.query}")
        print(f"答案: {rag.query(args.query, chain_of_thought=chain_of_thought)}\n")
    elif args.interactive:
        print(f"\n进入交互式问答模式（思维链: {'启用' if chain_of_thought else '关闭'}，输入 exit 或 quit 退出）\n")
        while True:
            try:
                question = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            answer = rag.query(question, chain_of_thought=chain_of_thought)
            print(answer)
            print()
    elif not args.query and not args.interactive:
        print(f"\nGraphRAG 已就绪！（思维链: {'启用' if chain_of_thought else '关闭'}，输入 exit 退出）\n")
        while True:
            try:
                question = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                break
            answer = rag.query(question, chain_of_thought=chain_of_thought)
            print(answer)
            print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "preprocess":
        run_preprocessing(args)
    elif args.command == "dataset":
        run_dataset(args)
    elif args.command == "rag-build":
        run_rag_build(args)
    elif args.command == "rag-query":
        run_rag_query(args)
    elif args.command == "rag":
        run_rag(args)
    else:
        # 无子命令时，默认行为（向后兼容）
        # 检查 --file / --query 等旧参数
        if any(hasattr(args, a) and getattr(args, a) for a in ["file", "query", "interactive"]):
            run_rag(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
