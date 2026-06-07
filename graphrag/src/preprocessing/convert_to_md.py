"""Docling 批量文档转换工具 —— 完全还原模式，最大程度保留原始文档内容。

支持:
  - 图片嵌入/引用（base64 内嵌或保存为文件）
  - 页眉页脚等装饰性内容
  - 分页标记
  - 表格、图表、公式、脚注等全部元素
  - 逐页或整文档导出
  - OCR 开关（对已有文本层的 PDF 可关闭 OCR 避免 RapidOCR 空检测警告）
  - 设备选择（CPU / CUDA），GPU OOM 时自动回退到 CPU

用法:
    python -m graphrag.src.preprocessing.convert_to_md                      # 默认转换 documents/
    python -m graphrag.src.preprocessing.convert_to_md --input docs/ --output docs_md/
    python -m graphrag.src.preprocessing.convert_to_md --file doc.pdf
    python -m graphrag.src.preprocessing.convert_to_md --image-mode embedded --per-page
    python -m graphrag.src.preprocessing.convert_to_md --no-ocr
    python -m graphrag.src.preprocessing.convert_to_md --force-ocr
    python -m graphrag.src.preprocessing.convert_to_md --device cpu
    python -m graphrag.src.preprocessing.convert_to_md --offline

转换后的 .md 文件可直接用 TxtLoader 加载进入 RAG 流水线。
"""

from __future__ import annotations

import argparse
import gc
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

_LOG = logging.getLogger(__name__)


def _now() -> str:
    """返回当前时间字符串 [HH:MM:SS]"""
    return datetime.now().strftime("%H:%M:%S")


def _cleanup_gpu_memory() -> None:
    """主动清理 GPU 缓存和 Python 垃圾，防止 GPU OOM。"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _gpu_memory_used_ratio() -> float:
    """返回当前 GPU 显存使用率 (0.0 ~ 1.0)，无 GPU 时返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            return 1.0 - free / total if total > 0 else 0.0
    except (ImportError, RuntimeError, AttributeError):
        pass
    return 0.0


def _ensure_gpu_memory(converter=None, threshold: float = 0.85) -> None:
    """如果 GPU 显存使用率超过阈值，强制清理并清除 converter 的 pipeline 缓存。

    Args:
        converter: DocumentConverter 实例，传入后将清除其内部 pipeline 缓存。
        threshold: 显存使用率阈值 (默认 0.85)。
    """
    ratio = _gpu_memory_used_ratio()
    if ratio < threshold:
        return
    _LOG.info("  GPU 显存使用率 %.0f%%，执行深度清理...", ratio * 100)
    if converter is not None:
        try:
            pipelines = converter._get_initialized_pipelines()
            old_count = len(pipelines)
            pipelines.clear()
            if old_count > 0:
                _LOG.info("  已清除 %d 个 pipeline 缓存", old_count)
        except Exception:
            pass
    _cleanup_gpu_memory()
    # 深度清理后再次检查
    ratio = _gpu_memory_used_ratio()
    if ratio >= threshold:
        _LOG.warning("  深度清理后显存仍高 (%.0f%%)，建议释放其他 GPU 进程", ratio * 100)


def _build_converter(
    do_ocr: bool,
    force_ocr: bool,
    device: str,
    generate_picture_images: bool,
) -> "DocumentConverter":
    """构建并返回一个 DocumentConverter 实例（用于复用，避免重复加载模型）。"""
    from docling.document_converter import DocumentConverter, FormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend

    pipe_opts = PdfPipelineOptions(
        do_ocr=do_ocr,
        force_backend_text=True,
        generate_picture_images=generate_picture_images,
    )
    if force_ocr:
        pipe_opts.do_ocr = True
        pipe_opts.force_backend_text = False
        from docling.datamodel.pipeline_options import RapidOcrOptions
        pipe_opts.ocr_options = RapidOcrOptions(
            force_full_page_ocr=True,
            bitmap_area_threshold=0.1,
        )

    if not do_ocr and not force_ocr:
        pipe_opts.force_backend_text = True

    actual_device = _setup_device_env(device)
    try:
        from docling.datamodel.pipeline_options import AcceleratorDevice
        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cuda": AcceleratorDevice.CUDA,
            "cpu": AcceleratorDevice.CPU,
        }
        pipe_opts.accelerator_options.device = device_map.get(
            actual_device, AcceleratorDevice.AUTO
        )
    except (ImportError, AttributeError):
        pass
    _LOG.info("  设备: %s | OCR: %s", actual_device, "强制" if force_ocr else ("开启" if do_ocr else "关闭"))

    fmt = FormatOption(
        pipeline_cls=StandardPdfPipeline,
        backend=DoclingParseDocumentBackend,
        pipeline_options=pipe_opts,
    )
    return DocumentConverter(format_options={InputFormat.PDF: fmt})


def _setup_device_env(device: str) -> str:
    """解析设备参数，返回实际设备名。"""
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _enable_offline_mode() -> None:
    """启用离线模式：跳过 HuggingFace Hub 联网验证，使用本地缓存。"""
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    _LOG.info("  离线模式: 使用本地缓存的模型（跳过 Hub 联网验证）")


def _setup_logger(log_file: str | None = None) -> str:
    """配置 convert_to_md 专用的文件日志（不干涉 root logger）。

    控制台输出由中央 logging 系统（setup_logging）负责。
    """
    if log_file is None:
        log_file = f"logs/convert_to_md_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    ))
    _LOG.addHandler(fh)

    _LOG.info("日志文件: %s", log_file)
    return log_file


def _split_pdf(src: Path, max_pages: int, temp_dir: Path) -> List[Path]:
    """用 pypdf 将大 PDF 拆分为多个小 PDF 块。"""
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(src))
    total = len(reader.pages)
    chunks: List[Path] = []

    for start in range(0, total, max_pages):
        end = min(start + max_pages, total)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        chunk_path = temp_dir / f"{src.stem}_p{start + 1:03d}-p{end:03d}.pdf"
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append(chunk_path)
        _LOG.info("  拆分块: %s (%d 页)", chunk_path.name, end - start)

    return chunks


def _output_exists(src: Path, output_dir: Path, per_page: bool) -> bool:
    """检查目标文件是否已完整输出（用于 --resume 跳过）。"""
    if per_page:
        return False
    return (output_dir / f"{src.stem}.md").exists()


def convert_file(
    file_path: str,
    output_dir: str,
    per_page: bool = False,
    image_mode: str = "embedded",
    page_breaks: bool = True,
    compact_tables: bool = False,
    include_furniture: bool = True,
    do_ocr: bool = True,
    force_ocr: bool = False,
    max_pages: int = 20,
    resume: bool = True,
    device: str = "auto",
    generate_picture_images: bool = True,
) -> List[str]:
    """用 Docling 将单个文件转为 Markdown 并保存（完全还原）。

    Args:
        file_path: 源文件路径。
        output_dir: 输出目录。
        per_page: True=每页一个 .md 文件, False=整个文档一个 .md 文件。
        image_mode: 图片处理方式 ("placeholder", "embedded", "referenced")。
        page_breaks: 是否在 MD 中标记分页。
        compact_tables: 是否使用紧凑表格格式。
        include_furniture: 是否包含页眉页脚等装饰性内容。
        do_ocr: 是否启用 OCR。
        force_ocr: 是否强制全页 OCR。
        max_pages: 大 PDF 拆分页数阈值（0=不拆分）。
        resume: 是否跳过已存在的输出。
        device: 推理设备 ("auto", "cuda", "cpu")。
        generate_picture_images: 是否生成图片。

    Returns:
        生成的 .md 文件路径列表。
    """
    for logger_name in ("rapidocr", "docling", "docling.backend.msword_backend"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # 创建共享的 DocumentConverter（复用到所有 chunk，避免重复加载 OCR 模型）
    converter = _build_converter(
        do_ocr=do_ocr,
        force_ocr=force_ocr,
        device=device,
        generate_picture_images=generate_picture_images,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)

    if resume and _output_exists(src, output_dir, per_page):
        _LOG.info("  [SKIP] %s 输出已存在，跳过", src.name)
        if per_page:
            return sorted(str(p) for p in output_dir.glob(f"{src.stem}_p*.md"))
        return [str(output_dir / f"{src.stem}.md")]

    # ---- 大 PDF 拆分逻辑 ----
    if max_pages > 0 and src.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(src))
            total_pages = len(reader.pages)
            if total_pages > max_pages:
                import tempfile

                temp_dir = Path(tempfile.mkdtemp(prefix="pdf_chunks_"))
                chunk_files = _split_pdf(src, max_pages, temp_dir)
                _LOG.info("  已拆分为 %d 个块 (最多 %d 页/块)，总页数: %d", len(chunk_files), max_pages, total_pages)

                generated: List[str] = []
                global_page_offset = 0
                for chunk_path in chunk_files:
                    if resume and per_page:
                        chunk_pages = _count_pages_in_pdf(chunk_path)
                        existing = set(output_dir.glob(f"{src.stem}_p*.md"))
                        all_done = all(
                            (output_dir / f"{src.stem}_p{(global_page_offset + p):03d}.md").exists()
                            for p in range(1, chunk_pages + 1)
                        )
                        if all_done:
                            _LOG.info("  [SKIP] 块 %s 页面已存在，跳过", chunk_path.name)
                            global_page_offset += chunk_pages
                            for p in range(1, chunk_pages + 1):
                                generated.append(
                                    str(output_dir / f"{src.stem}_p{(global_page_offset - chunk_pages + p):03d}.md")
                                )
                            continue

                    if resume and not per_page:
                        chunk_md = output_dir / f"{chunk_path.stem}.md"
                        if chunk_md.exists():
                            _LOG.info("  [SKIP] 块 %s 已转换，跳过", chunk_path.name)
                            generated.append(str(chunk_md))
                            continue

                    # 检查 GPU 显存，不足则直接走 CPU
                    chunk_device = device
                    chunk_conv = converter
                    if device != "cpu" and _gpu_memory_used_ratio() > 0.80:
                        _LOG.info("  GPU 显存使用率 %.0f%%，此块走 CPU", _gpu_memory_used_ratio() * 100)
                        chunk_device = "cpu"
                        chunk_conv = None

                    try:
                        chunk_result = _convert_single(
                            str(chunk_path), output_dir, per_page, image_mode,
                            page_breaks, compact_tables, include_furniture,
                            do_ocr, force_ocr, global_page_offset, device=chunk_device,
                            resume=resume,
                            generate_picture_images=generate_picture_images,
                            converter=chunk_conv,
                        )
                    except (RuntimeError, MemoryError) as _e:
                        err_str = str(_e).lower()
                        if chunk_device != "cpu" and ("out of memory" in err_str or "cuda" in err_str or "cublas" in err_str):
                            _LOG.warning("  [OOM] GPU 内存不足，chunk 切换到 CPU 重试...")
                            _cleanup_gpu_memory()
                            chunk_result = _convert_single(
                                str(chunk_path), output_dir, per_page, image_mode,
                                page_breaks, compact_tables, include_furniture,
                                do_ocr, force_ocr, global_page_offset, device="cpu",
                                resume=resume,
                                generate_picture_images=generate_picture_images,
                                converter=None,
                            )
                        elif chunk_device == "cpu" and ("out of memory" in err_str or "cuda" in err_str):
                            _LOG.warning("  [OOM] CPU 模式也出现 OOM，跳过此块...")
                            raise
                        else:
                            raise
                    generated.extend(chunk_result)
                    if per_page:
                        global_page_offset += _count_pages_in_pdf(chunk_path)

                    _ensure_gpu_memory(converter)

                if not per_page:
                    full_md_parts = []
                    for md_path in generated:
                        full_md_parts.append(Path(md_path).read_text(encoding="utf-8"))
                    combined_md = f"\n\n--- pagebreak ---\n\n".join(full_md_parts)
                    md_name = f"{src.stem}.md"
                    md_path = output_dir / md_name
                    md_path.write_text(combined_md.strip(), encoding="utf-8")
                    for md_path_old in generated:
                        Path(md_path_old).unlink(missing_ok=True)
                    generated = [str(md_path)]
                    _LOG.info("  [OK] %s (合并 %d 页)", md_path.name, total_pages)

                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                del converter
                _cleanup_gpu_memory()
                return generated
        except Exception as e:
            import traceback
            _LOG.warning("  [WARN] PDF 拆分失败 (%s: %s)，将尝试直接转换", type(e).__name__, e)
            traceback.print_exc()

    cpu_converter = None
    try:
        return _convert_single(
            file_path, output_dir, per_page, image_mode,
            page_breaks, compact_tables, include_furniture,
            do_ocr, force_ocr, device=device, resume=resume,
            generate_picture_images=generate_picture_images,
            converter=converter,
        )
    except (RuntimeError, MemoryError) as e:
        err_str = str(e).lower()
        if device != "cpu" and ("out of memory" in err_str or "cuda" in err_str or "cublas" in err_str):
            _LOG.warning("  [OOM] GPU 内存不足 (%s)，自动切换到 CPU 重试...", type(e).__name__)
            _cleanup_gpu_memory()
            cpu_converter = _build_converter(do_ocr, force_ocr, "cpu", generate_picture_images)
            return _convert_single(
                file_path, output_dir, per_page, image_mode,
                page_breaks, compact_tables, include_furniture,
                do_ocr, force_ocr, device="cpu", resume=resume,
                generate_picture_images=generate_picture_images,
                converter=cpu_converter,
            )
        raise
    except Exception as e:
        import traceback
        _LOG.error("  [FAIL] %s: %s: %s", src.name, type(e).__name__, e)
        traceback.print_exc()
        raise
    finally:
        del converter
        if cpu_converter is not None:
            del cpu_converter
        _cleanup_gpu_memory()


def _convert_single(
    file_path: str,
    output_dir: Path,
    per_page: bool,
    image_mode: str,
    page_breaks: bool,
    compact_tables: bool,
    include_furniture: bool,
    do_ocr: bool,
    force_ocr: bool,
    page_offset: int = 0,
    device: str = "auto",
    resume: bool = True,
    generate_picture_images: bool = True,
    converter: Optional["DocumentConverter"] = None,
) -> List[str]:
    """用 Docling 转换单个 PDF/DOCX 等文件，并导出 Markdown。

    Args:
        converter: 可复用的 DocumentConverter 实例。为 None 时自动创建。
    """
    from docling_core.types.doc.base import ImageRefMode

    src = Path(file_path)
    converter_owned = converter is None
    if converter is None:
        converter = _build_converter(do_ocr, force_ocr, device, generate_picture_images)
    _LOG.info("  设备信息参见上方 | 文件: %s", src.name)
    result = converter.convert(source=str(src))
    doc = result.document

    img_mode = {
        "placeholder": ImageRefMode.PLACEHOLDER,
        "embedded": ImageRefMode.EMBEDDED,
        "referenced": ImageRefMode.REFERENCED,
    }.get(image_mode, ImageRefMode.EMBEDDED)

    export_kwargs: dict = dict(
        image_mode=img_mode,
        escape_html=True,
        escape_underscores=True,
        enable_chart_tables=True,
        include_annotations=True,
        mark_annotations=False,
        compact_tables=compact_tables,
        traverse_pictures=True,
    )

    if include_furniture:
        from docling_core.types.doc.document import ContentLayer
        export_kwargs["included_content_layers"] = {
            ContentLayer.BODY,
            ContentLayer.FURNITURE,
            ContentLayer.NOTES,
        }

    if page_breaks:
        export_kwargs["page_break_placeholder"] = "\n\n--- pagebreak ---\n\n"

    generated: List[str] = []

    if per_page:
        for page_num in sorted(doc.pages.keys()):
            global_page = page_offset + page_num
            md_name = f"{src.stem}_p{global_page:03d}.md"
            md_path = output_dir / md_name

            if resume and md_path.exists():
                generated.append(str(md_path))
                continue

            page_md = doc.export_to_markdown(page_no=page_num, **export_kwargs)
            md_path.write_text(page_md.strip(), encoding="utf-8")
            generated.append(str(md_path))
            _LOG.info("  [OK] %s", md_path.name)
    else:
        full_md = doc.export_to_markdown(**export_kwargs)
        md_name = f"{src.stem}.md"
        md_path = output_dir / md_name
        md_path.write_text(full_md.strip(), encoding="utf-8")
        generated.append(str(md_path))
        _LOG.info("  [OK] %s", md_path.name)

    del doc, result
    if converter_owned:
        del converter
    _cleanup_gpu_memory()

    return generated


def _count_pages_in_pdf(path: Path) -> int:
    """快速获取 PDF 页数。"""
    from pypdf import PdfReader
    return len(PdfReader(str(path)).pages)


def validate_md_files(
    output_dir: str,
    input_dir: str | None = None,
) -> List[tuple]:
    """扫描输出目录，检查每个 .md 文件的质量问题。"""
    out_path = Path(output_dir)
    if not out_path.is_dir():
        _LOG.warning("输出目录不存在: %s", output_dir)
        return []

    issues: List[tuple] = []
    md_files = sorted(out_path.glob("*.md"))

    if not md_files:
        _LOG.info("输出目录中没有 .md 文件: %s", output_dir)
        return []

    _LOG.info("正在验证 %d 个 .md 文件...", len(md_files))

    import re

    for md_path in md_files:
        size = md_path.stat().st_size

        if size == 0:
            issues.append((md_path, "ERROR", "文件为空 (0 bytes)"))
            continue

        if size < 1024:
            issues.append((md_path, "ERROR", f"文件过小 ({size} bytes)，可能转换失败"))
            continue
        if size < 5 * 1024:
            issues.append((md_path, "WARNING", f"文件偏小 ({size} bytes)，可能不完整"))

        content = md_path.read_text(encoding="utf-8", errors="replace")
        if "Image not available" in content:
            issues.append((md_path, "WARNING", "包含 'Image not available'"))

    if input_dir:
        in_path = Path(input_dir)
        for md_path in md_files:
            stem = md_path.stem
            base_stem = re.sub(r"_p\d{3}(?:-\d{3})?$", "", stem)
            found = False
            for ext in [".pdf", ".docx", ".pptx", ".html", ".htm", ".doc", ".xlsx"]:
                candidates = list(in_path.rglob(f"{base_stem}{ext}"))
                if candidates:
                    found = True
                    break
            if not found:
                issues.append((md_path, "WARNING", f"未找到对应的输入文件 (stem={base_stem})"))

    errors = sum(1 for _, sev, _ in issues if sev == "ERROR")
    warnings = sum(1 for _, sev, _ in issues if sev == "WARNING")
    _LOG.info(
        "验证完成: %d 个问题 (ERROR: %d, WARNING: %d)",
        len(issues), errors, warnings,
    )

    if issues:
        _LOG.info("问题详情:")
        for fp, sev, reason in issues:
            _LOG.info("  [%s] %s: %s", sev, fp.name, reason)

    return issues


def _find_input_for_output(
    md_path: Path,
    input_dir: str,
) -> Path | None:
    """根据输出 .md 文件的路径，在输入目录中查找对应的源文件。"""
    import re
    in_path = Path(input_dir)
    stem = md_path.stem
    base_stem = re.sub(r"_p\d{3}(?:-p?\d{3})?$", "", stem)

    for ext in [".pdf", ".docx", ".pptx", ".html", ".htm", ".doc", ".xlsx"]:
        candidates = sorted(in_path.rglob(f"{base_stem}{ext}"))
        if candidates:
            return candidates[0]

    for ext in [".pdf", ".docx", ".pptx", ".html", ".htm", ".doc", ".xlsx"]:
        direct = in_path / f"{stem}{ext}"
        if direct.exists():
            return direct

    return None


def reconvert_bad_files(
    issues: List[tuple],
    input_dir: str,
    output_dir: str,
    per_page: bool = False,
    image_mode: str = "embedded",
    page_breaks: bool = True,
    compact_tables: bool = False,
    include_furniture: bool = True,
    do_ocr: bool = True,
    force_ocr: bool = False,
    max_pages: int = 50,
    resume: bool = True,
    device: str = "auto",
) -> int:
    """重新转换有问题的文件。"""
    reconverted = 0
    for md_path, severity, reason in issues:
        src = _find_input_for_output(md_path, input_dir)
        if src is None:
            _LOG.warning("  [SKIP] 找不到 %s 对应的源文件", md_path.name)
            continue

        gen_pic = "Image not available" in reason
        if gen_pic:
            _LOG.info(
                "  [RECONVERT] %s (原因: %s, 启用 generate_picture_images=True)",
                src.name, reason,
            )
        else:
            _LOG.info("  [RECONVERT] %s (原因: %s)", src.name, reason)

        md_path.unlink(missing_ok=True)

        try:
            convert_file(
                str(src), output_dir,
                per_page=per_page, image_mode=image_mode,
                page_breaks=page_breaks, compact_tables=compact_tables,
                include_furniture=include_furniture,
                do_ocr=do_ocr, force_ocr=force_ocr,
                max_pages=max_pages, resume=False, device=device,
                generate_picture_images=gen_pic,
            )
            reconverted += 1
        except Exception as e:
            import traceback
            _LOG.error("  [FAIL] 重新转换 %s 失败: %s: %s", src.name, type(e).__name__, e)
            traceback.print_exc()

    _LOG.info("重新转换完成: %d/%d 个文件", reconverted, len(issues))
    return reconverted


def batch_convert(
    input_dir: str,
    output_dir: str,
    formats: Optional[List[str]] = None,
    per_page: bool = False,
    image_mode: str = "embedded",
    page_breaks: bool = True,
    compact_tables: bool = False,
    include_furniture: bool = True,
    do_ocr: bool = True,
    force_ocr: bool = False,
    max_pages: int = 50,
    resume: bool = True,
    device: str = "auto",
    generate_picture_images: bool = True,
    validate: bool = True,
    reconvert_bad: bool = False,
) -> int:
    """批量转换目录下所有文档为 Markdown（完全还原）。"""
    if formats is None:
        formats = [".pdf", ".docx", ".pptx", ".html", ".htm"]

    input_path = Path(input_dir)
    files = []
    for ext in formats:
        files.extend(input_path.rglob(f"*{ext}"))

    if not files:
        _LOG.info("在 %s 中未找到 %s 格式的文件", input_dir, formats)
        return 0

    _LOG.info("找到 %d 个文件，开始转换...\n", len(files))
    cleanup_interval = max(1, min(5, len(files) // 10 + 1))
    success = 0
    for idx, fp in enumerate(sorted(files)):
        if idx > 0 and idx % cleanup_interval == 0:
            _cleanup_gpu_memory()

        if resume and _output_exists(fp, Path(output_dir), per_page):
            _LOG.info("[%s] [SKIP] %s 输出已存在，跳过", _now(), fp.name)
            success += 1
            continue
        try:
            _LOG.info("[%s] (%d/%d) 转换: %s", _now(), idx + 1, len(files), fp.name)
            convert_file(
                str(fp), output_dir,
                per_page=per_page, image_mode=image_mode,
                page_breaks=page_breaks, compact_tables=compact_tables,
                include_furniture=include_furniture,
                do_ocr=do_ocr, force_ocr=force_ocr,
                max_pages=max_pages, resume=resume,
                device=device, generate_picture_images=generate_picture_images,
            )
            success += 1
        except (RuntimeError, MemoryError) as e:
            err_str = str(e).lower()
            if device != "cpu" and ("out of memory" in err_str or "cuda" in err_str):
                _LOG.warning("[%s] [OOM] %s GPU 内存不足，清空缓存后重试...", _now(), fp.name)
                _cleanup_gpu_memory()
                try:
                    convert_file(
                        str(fp), output_dir,
                        per_page=per_page, image_mode=image_mode,
                        page_breaks=page_breaks, compact_tables=compact_tables,
                        include_furniture=include_furniture,
                        do_ocr=do_ocr, force_ocr=force_ocr,
                        max_pages=max_pages, resume=resume,
                        device="cpu",
                        generate_picture_images=generate_picture_images,
                    )
                    success += 1
                except Exception as e2:
                    _LOG.error("[%s] [FAIL] %s: %s: %s", _now(), fp.name, type(e2).__name__, e2)
                    import traceback
                    traceback.print_exc()
            else:
                _LOG.error("[%s] [FAIL] %s: %s: %s", _now(), fp.name, type(e).__name__, e)
                import traceback
                traceback.print_exc()
        except Exception as e:
            import traceback
            _LOG.error("[%s] [FAIL] %s: %s: %s", _now(), fp.name, type(e).__name__, e)
            traceback.print_exc()

    _LOG.info("完成: %d/%d 个文件转换成功", success, len(files))
    _LOG.info("输出目录: %s", output_dir)

    if validate:
        issues = validate_md_files(output_dir, input_dir=input_dir)
        if issues:
            issues_warn = [i for i in issues if i[1] == "WARNING"]
            issues_err = [i for i in issues if i[1] == "ERROR"]
            _LOG.info(
                "验证结果: %d 个问题 (ERROR: %d, WARNING: %d)",
                len(issues), len(issues_err), len(issues_warn),
            )
            if reconvert_bad:
                _LOG.info("开始重新转换 %d 个有问题的文件...", len(issues))
                reconvert_bad_files(
                    issues, input_dir, output_dir,
                    per_page=per_page, image_mode=image_mode,
                    page_breaks=page_breaks, compact_tables=compact_tables,
                    include_furniture=include_furniture,
                    do_ocr=do_ocr, force_ocr=force_ocr,
                    max_pages=max_pages, resume=resume, device=device,
                )
                _LOG.info("重新验证...")
                issues2 = validate_md_files(output_dir, input_dir=input_dir)
                remaining = len(issues2)
                if remaining == 0:
                    _LOG.info("所有问题已修复！")
                else:
                    _LOG.info("仍有 %d 个问题未解决。", remaining)
            else:
                _LOG.info("N 个文件有问题。使用 --reconvert-bad 重新转换。")
        else:
            _LOG.info("验证通过：所有文件正常。")

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Docling 文档转 Markdown 工具（完全还原模式）"
    )
    parser.add_argument("--input", default=None,
                        help="输入目录（默认 documents/）")
    parser.add_argument("--output", default=None,
                        help="输出目录（默认 documents_md/）")
    parser.add_argument("--file", default=None,
                        help="转换单个文件")
    parser.add_argument("--per-page", action="store_true",
                        help="每页生成一个 .md 文件")
    parser.add_argument("--image-mode",
                        choices=["placeholder", "embedded", "referenced"],
                        default="embedded",
                        help="图片处理方式：placeholder=占位符, embedded=base64内嵌（默认）, referenced=引用保存")
    parser.add_argument("--no-page-breaks", action="store_true",
                        help="不标记分页")
    parser.add_argument("--compact-tables", action="store_true",
                        help="使用紧凑表格格式")
    parser.add_argument("--no-furniture", action="store_true",
                        help="不包含页眉页脚等装饰性内容")
    parser.add_argument("--no-ocr", action="store_true",
                        help="关闭 OCR（适合已有文本层的数字 PDF）")
    parser.add_argument("--force-ocr", action="store_true",
                        help="强制全页 OCR（适合扫描件 PDF）")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"],
                        default="auto",
                        help="推理设备：auto=自动检测(默认), cuda=强制GPU, cpu=强制CPU")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="大 PDF 拆分页数阈值（默认 20，设为 0=不拆分）")
    parser.add_argument("--no-resume", action="store_false", dest="resume", default=True,
                        help="禁用断点续传（默认启用）")
    parser.add_argument("--offline", action="store_true",
                        help="离线模式：使用本地缓存的模型")
    parser.add_argument("--log-file", default=None,
                        help="日志文件路径")
    parser.add_argument("--validate", action="store_true",
                        help="仅验证输出目录中的 .md 文件质量（不转换）")
    parser.add_argument("--reconvert-bad", action="store_false",
                        help="验证后自动重新转换有问题的文件")
    parser.add_argument("--no-validate", action="store_true",
                        help="转换完成后跳过验证")
    args = parser.parse_args()

    if args.offline:
        _enable_offline_mode()
    log_path = _setup_logger(args.log_file)
    _LOG.info("命令行参数: %s", vars(args))

    convert_kwargs = dict(
        image_mode=args.image_mode,
        page_breaks=not args.no_page_breaks,
        compact_tables=args.compact_tables,
        include_furniture=not args.no_furniture,
        do_ocr=not args.no_ocr,
        force_ocr=args.force_ocr,
        max_pages=args.max_pages,
        resume=args.resume,
        device=args.device,
        generate_picture_images=True,
    )

    if args.file:
        src = Path(args.file)
        out_dir = args.output or f"{src.stem}_md"
        _LOG.info("转换单个文件: %s", src)
        try:
            convert_file(str(src), out_dir, per_page=args.per_page, **convert_kwargs)
        except Exception as e:
            import traceback
            _LOG.error("[FAIL] %s: %s: %s", src.name, type(e).__name__, e)
            traceback.print_exc()
            return
        _LOG.info("输出目录: %s", out_dir)
        return

    input_dir = args.input or str(
        Path(__file__).resolve().parent.parent.parent.parent / "documents"
    )
    output_dir = args.output or str(
        Path(input_dir).parent / f"{Path(input_dir).name}_md"
    )

    if args.validate:
        _LOG.info("验证模式: 检查 %s", output_dir)
        issues = validate_md_files(str(output_dir), input_dir=str(input_dir))
        if not issues:
            _LOG.info("验证通过：所有文件正常。")
        elif args.reconvert_bad:
            _LOG.info("开始重新转换 %d 个有问题的文件...", len(issues))
            reconvert_bad_files(
                issues, str(input_dir), str(output_dir),
                per_page=args.per_page, **convert_kwargs,
            )
            _LOG.info("重新验证...")
            issues2 = validate_md_files(str(output_dir), input_dir=str(input_dir))
            remaining = len(issues2)
            if remaining == 0:
                _LOG.info("所有问题已修复！")
            else:
                _LOG.info("仍有 %d 个问题未解决。", remaining)
        return

    batch_convert(
        str(input_dir), str(output_dir),
        per_page=args.per_page,
        validate=not args.no_validate,
        reconvert_bad=args.reconvert_bad,
        **convert_kwargs,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        _LOG.error("[FATAL] 程序异常退出: %s: %s", type(e).__name__, e)
        traceback.print_exc()
        import sys
        sys.exit(1)
