"""
PDF 处理工具函数
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger


def convert_pdf_to_images(pdf_path: Path, output_dir: Path, zoom: float = 2.0, dpi: Optional[int] = None) -> List[Path]:
    """
    将 PDF 所有页转换为图片

    Args:
        pdf_path: PDF 文件路径
        output_dir: 输出目录
        zoom: 缩放比例（默认 2.0，即 2 倍）
        dpi: DPI 设置（可选，如果设置则会覆盖 zoom）

    Returns:
        转换后的图片路径列表
    """
    try:
        import fitz  # PyMuPDF

        # 打开 PDF
        doc = fitz.open(str(pdf_path))

        # 获取页数
        page_count = len(doc)

        logger.info(f"📄 PDF has {page_count} pages")

        image_paths = []

        # 处理所有页面
        for page_num in range(page_count):
            page = doc[page_num]

            # 设置缩放/DPI
            if dpi:
                # 如果指定了 DPI，计算对应的缩放比例
                # 默认 PDF DPI 是 72
                zoom = dpi / 72.0

            mat = fitz.Matrix(zoom, zoom)

            # 渲染为图片
            pix = page.get_pixmap(matrix=mat)

            # 保存为 PNG（统一命名格式）
            image_path = output_dir / f"{pdf_path.stem}_page{page_num + 1}.png"

            pix.save(str(image_path))
            image_paths.append(image_path)

            logger.debug(f"   Converted page {page_num + 1}/{page_count} to PNG")

        # 关闭文档
        doc.close()

        logger.info(f"   Converted all {page_count} pages to PNG")

        return image_paths

    except ImportError:
        logger.error("❌ PyMuPDF not installed. Install with: pip install PyMuPDF")
        raise RuntimeError("PyMuPDF is required for PDF processing")
    except Exception as e:
        logger.error(f"❌ Failed to convert PDF to images: {e}")
        raise


def get_pdf_page_count(pdf_path: Path) -> int:
    """
    获取 PDF 文件的总页数

    Args:
        pdf_path: PDF 文件路径

    Returns:
        int: 页数
    """
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        logger.error(f"❌ Failed to get PDF page count: {e}")
        # 如果读取失败，返回 0 或抛出异常，视业务逻辑而定
        # 这里返回 0 让上层逻辑决定如何处理（通常是不拆分）
        return 0


def split_pdf_file(
    pdf_path: Path, 
    output_dir: Path, 
    chunk_size: int = 500, 
    parent_task_id: str = ""
) -> List[Dict[str, Any]]:
    """
    将大 PDF 文件拆分为多个小文件

    Args:
        pdf_path: 源 PDF 路径
        output_dir: 输出目录
        chunk_size: 每个分块的页数
        parent_task_id: 父任务 ID（用于日志或命名）

    Returns:
        List[Dict]: 分块信息列表，每个元素包含:
            - path: 分块文件路径
            - start_page: 起始页码 (1-based)
            - end_page: 结束页码 (1-based)
            - page_count: 该分块页数
    """
    try:
        import fitz  # PyMuPDF

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        chunks = []

        logger.info(f"✂️ Splitting PDF ({total_pages} pages) into chunks of {chunk_size}")

        # 计算分块
        for i in range(0, total_pages, chunk_size):
            start_page = i
            end_page = min(i + chunk_size, total_pages)
            
            # 创建新的 PDF 文档
            new_doc = fitz.open()
            
            # 插入页面 (from_page 是包含的, to_page 也是包含的，fitz 使用 0-based 索引)
            # insert_pdf 参数: from_page, to_page
            new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
            
            # 生成文件名: original_pages_1-500.pdf
            # 注意：对外文件名使用 1-based 索引，符合人类直觉
            chunk_filename = f"{pdf_path.stem}_pages_{start_page + 1}-{end_page}.pdf"
            chunk_path = output_dir / chunk_filename
            
            new_doc.save(str(chunk_path))
            new_doc.close()

            chunks.append({
                "path": str(chunk_path),
                "start_page": start_page + 1,  # 1-based
                "end_page": end_page,          # 1-based
                "page_count": end_page - start_page
            })

            logger.debug(f"   Created chunk: {chunk_filename} ({end_page - start_page} pages)")

        doc.close()
        return chunks

    except Exception as e:
        logger.error(f"❌ Failed to split PDF: {e}")
        raise
