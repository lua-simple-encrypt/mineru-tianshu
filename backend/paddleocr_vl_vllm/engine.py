"""
PaddleOCR-VL-VLLM 解析引擎 (全功能版)
单例模式，每个进程只加载一次基础版面识别模型,OCR部分调用配置的API
使用最新的 PaddleOCR-VL-VLLM API（自动多语言识别）

参考文档：https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html#322-python-api
"""

from pathlib import Path
from typing import Optional, Dict, Any
from threading import Lock
from loguru import logger
import json
import os


class PaddleOCRVLVLLMEngine:
    """
    PaddleOCR-VL-VLLM 解析引擎（新版本）

    特性：
    - 单例模式（每个进程只加载一次模型）
    - 自动多语言识别（无需指定语言，支持 109+ 语言）
    - 线程安全
    - 仅支持 GPU 推理（不支持 CPU）
    - 原生支持 PDF 多页文档解析
    - 结构化输出（Markdown/JSON）
    - 模型自动下载和缓存（由 PaddleOCR 管理，无需手动下载）

    GPU 要求：
    - NVIDIA GPU with Compute Capability ≥ 8.5
    - 推荐：RTX 3090, RTX 4090, A10, A100, H100
    """

    _instance: Optional["PaddleOCRVLVLLMEngine"] = None
    _lock = Lock()
    _pipeline = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, device: str = "cuda:0", vllm_api_base: str = "http://localhost:17300/v1"):
        """
        初始化引擎（只执行一次）

        Args:
            device: 设备 (cuda:0, cuda:1 等，PaddleOCR 仅支持 GPU)
            vllm_api_base: VLLM API 基础 URL (默认: http://localhost:17300/v1)
        """
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self.device = device  # 保存 device 参数
            self.vllm_api_base = vllm_api_base  # 保存 vllm_api_base 参数

            # 从 device 字符串中提取 GPU ID (例如 "cuda:0" -> 0)
            if "cuda:" in device:
                self.gpu_id = int(device.split(":")[-1])
            else:
                self.gpu_id = 0
                logger.warning(f"⚠️  Invalid device format: {device}, using GPU 0")

            # 检查 GPU 可用性（PaddleOCR-VL 仅支持 GPU）
            self._check_gpu_availability()

            self._initialized = True

            logger.info("🔧 PaddleOCR-VL-VLLM Engine initialized")
            logger.info(f"   Device: {self.device} (GPU ID: {self.gpu_id})")
            logger.info(f"   VLLM API Base: {self.vllm_api_base}")
            logger.info("   Model: PaddlePaddle/PaddleOCR-VL (auto-managed)")
            logger.info("   Auto Multi-Language: Enabled (109+ languages)")
            logger.info("   GPU Only: CPU not supported")
            logger.info("   Model Cache: ~/.paddleocr/models/ (auto-managed)")

    def _check_gpu_availability(self):
        """
        检查 GPU 信息并输出日志
        PaddleOCR-VL 仅支持 GPU 推理，但不阻止低版本 GPU 运行
        """
        try:
            import paddle

            # 检查是否编译了 CUDA 支持
            if not paddle.is_compiled_with_cuda():
                logger.warning("⚠️  PaddlePaddle is not compiled with CUDA")
                logger.warning("   PaddleOCR-VL requires GPU support")
                logger.warning("   Install: pip install paddlepaddle-gpu==3.2.0")
                return

            # 检查是否有可用的 GPU
            gpu_count = paddle.device.cuda.device_count()
            if gpu_count == 0:
                logger.warning("⚠️  No CUDA devices found")
                logger.warning("   PaddleOCR-VL requires GPU for inference")
                return

            # 获取 GPU 信息
            try:
                gpu_name = paddle.device.cuda.get_device_name(0)
                compute_capability = paddle.device.cuda.get_device_capability(0)

                logger.info(f"✅ GPU detected: {gpu_name}")
                logger.info(f"   Compute Capability: {compute_capability[0]}.{compute_capability[1]}")
                logger.info(f"   GPU Count: {gpu_count}")

                # 仅输出建议，不阻止运行
                cc_major = compute_capability[0]
                cc_minor = compute_capability[1]
                if cc_major < 8 or (cc_major == 8 and cc_minor < 5):
                    logger.info("ℹ️  GPU Compute Capability < 8.5")
                    logger.info("   Official recommendation: CC ≥ 8.5 for best performance")
                    logger.info("   Your GPU may still work, but performance might vary")
            except Exception as e:
                logger.debug(f"Could not get detailed GPU info: {e}")

        except ImportError:
            logger.warning("⚠️  PaddlePaddle not installed")
            logger.warning("   Install: pip install paddlepaddle-gpu==3.2.0")
        except Exception as e:
            logger.debug(f"GPU check warning: {e}")

    def _load_pipeline(self):
        """延迟加载 PaddleOCR-VL-VLLM 管道"""
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            logger.info("=" * 60)
            logger.info("📥 Loading PaddleOCR-VL-VLLM Pipeline into memory...")
            logger.info("=" * 60)

            try:
                import paddle
                from paddleocr import PaddleOCRVL

                # 设置 PaddlePaddle 使用指定的 GPU
                # 必须在创建 PaddleOCRVL 实例之前设置
                if paddle.is_compiled_with_cuda():
                    paddle.set_device(f"gpu:{self.gpu_id}")
                    logger.info(f"🎯 PaddlePaddle device set to: gpu:{self.gpu_id}")
                else:
                    logger.warning("⚠️  CUDA not available, PaddleOCR-VL may not work")

                # 初始化 PaddleOCR-VL（新版本 API）
                logger.info("🤖 Initializing PaddleOCR-VL-VLLM Pipeline...")
                
                if self.vllm_api_base is None:
                    raise ValueError(
                        "vllm_api_base 不能为 None，请检查paddleocr-vl-vllm-engine-enabled 及 paddleocr-vl-vllm-api-list 配置"
                    )
                else:
                    # 初始化 pipeline
                    # 注意：这里仅做基础初始化，具体的功能开关（如印章、矫正）在 predict 时通过参数控制
                    self._pipeline = PaddleOCRVL(
                        vl_rec_backend="vllm-server",  # 使用 VLLM 后端
                        vl_rec_server_url=self.vllm_api_base,  # VLLM 服务器地址
                        use_layout_detection=True  # 默认开启基础版面分析
                    )

                logger.info("=" * 60)
                logger.info("✅ PaddleOCR-VL-VLLM Pipeline loaded successfully!")
                logger.info(f"   Device: GPU {self.gpu_id}")
                logger.info("   Backend: VLLM Server")
                logger.info("=" * 60)

                return self._pipeline

            except Exception as e:
                logger.error("=" * 80)
                logger.error("❌ 管道加载失败:")
                logger.error(f"   错误类型: {type(e).__name__}")
                logger.error(f"   错误信息: {e}")
                logger.error("=" * 80)

                import traceback
                logger.debug("完整堆栈跟踪:")
                logger.debug(traceback.format_exc())
                raise

    def cleanup(self):
        """
        清理推理产生的显存（不卸载模型）
        """
        try:
            import paddle
            import gc

            # 清理 PaddlePaddle 显存
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
                logger.debug("🧹 PaddleOCR-VL-VLLM: CUDA cache cleared")

            # 清理 Python 对象
            gc.collect()

            logger.debug("🧹 PaddleOCR-VL-VLLM: Memory cleanup completed")
        except Exception as e:
            logger.debug(f"Memory cleanup warning: {e}")

    def parse(self, file_path: str, output_path: str, **kwargs) -> Dict[str, Any]:
        """
        全功能解析入口：解析文档或图片

        Args:
            file_path: 输入文件路径
            output_path: 输出目录
            **kwargs: 动态接收官网支持的所有高级参数，例如：
                - use_doc_orientation_classify (bool): 图片方向矫正
                - use_doc_unwarping (bool): 图片扭曲矫正
                - use_seal_recognition (bool): 印章识别
                - use_chart_recognition (bool): 图表识别
                - use_ocr_for_image_block (bool): 图片文字识别
                - merge_tables (bool): 跨页表格合并 (后处理)
                - relevel_titles (bool): 段落标题级别识别 (后处理)
                - markdown_ignore_labels (list): 辅助内容过滤 (如页眉页脚)
                - layout_shape_mode (str): 版面形状 (auto/rect/quad/poly)
                - min_pixels, max_pixels (int): 图像像素限制
                - repetition_penalty, temperature, top_p (float): VLLM 生成参数

        Returns:
            解析结果（同时保存 Markdown 和 JSON 两种格式）
        """
        file_path = Path(file_path)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"🤖 PaddleOCR-VL-VLLM parsing: {file_path.name}")
        
        # 加载管道
        pipeline = self._load_pipeline()

        # 1. 构造 predict 参数字典
        # 使用 kwargs.get() 设置默认值，确保与官网 API 默认行为一致
        predict_params = {
            "input": str(file_path),
            
            # --- 图像矫正 & 预处理 ---
            "use_doc_orientation_classify": kwargs.get("use_doc_orientation_classify", False),
            "use_doc_unwarping": kwargs.get("use_doc_unwarping", False),
            "min_pixels": kwargs.get("min_pixels", 147384),
            "max_pixels": kwargs.get("max_pixels", 2822400),
            
            # --- 版面分析 & 识别功能 ---
            "use_layout_detection": kwargs.get("use_layout_detection", True),
            "use_chart_recognition": kwargs.get("use_chart_recognition", False),
            "use_seal_recognition": kwargs.get("use_seal_recognition", False),
            "use_ocr_for_image_block": kwargs.get("use_ocr_for_image_block", False),
            
            # --- 高级设置 ---
            "layout_shape_mode": kwargs.get("layout_shape_mode", "auto"), # auto, rect, quad, poly
            "layout_nms": kwargs.get("layout_nms", True),
            "prompt_label": kwargs.get("prompt_label", None), # 仅当 use_layout_detection=False 时生效
            
            # --- VLLM 生成参数 ---
            "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
            "temperature": kwargs.get("temperature", 0.0),
            "top_p": kwargs.get("top_p", 1.0),
            
            # --- 辅助内容过滤 (Markdown忽略标签) ---
            # 默认忽略：页码(number), 脚注(footnote), 页眉(header), 页脚(footer)等
            "markdown_ignore_labels": kwargs.get("markdown_ignore_labels", [
                'number', 'footnote', 'header', 'header_image', 
                'footer', 'footer_image', 'aside_text'
            ]),
        }
        
        # 打印关键参数以便调试
        logger.info(f"⚙️  功能开关: 方向矫正={predict_params['use_doc_orientation_classify']}, "
                    f"扭曲矫正={predict_params['use_doc_unwarping']}, "
                    f"印章识别={predict_params['use_seal_recognition']}")

        # 执行推理
        try:
            # 2. 调用 Pipeline 进行预测
            result = pipeline.predict(**predict_params)
            logger.info("✅ 推理完成")

            # 3. 后处理：页面重构 (跨页合并、标题分级)
            # 这些功能是通过 restructure_pages 实现的
            should_restructure = kwargs.get("restructure_pages", True) # 默认开启
            
            if should_restructure and hasattr(pipeline, "restructure_pages"):
                logger.info("🔄 正在执行页面重构 (表格合并 & 标题分级)...")
                try:
                    result = pipeline.restructure_pages(
                        result,
                        merge_table=kwargs.get("merge_tables", True),     # 跨页表格合并
                        relevel_titles=kwargs.get("relevel_titles", True) # 标题级别识别
                    )
                    logger.info("✅ 页面重构完成")
                except Exception as re_err:
                    logger.warning(f"⚠️ 页面重构失败 (降级使用原始结果): {re_err}")
                    import traceback
                    logger.debug(traceback.format_exc())

            logger.info(f"   识别了 {len(result)} 页/张")

            # 4. 保存结果
            markdown_list = []
            json_list = []

            for idx, res in enumerate(result, 1):
                logger.info(f"📝 处理结果 {idx}/{len(result)}")

                try:
                    # 为每页创建子目录并保存完整结果（便于调试）
                    page_output_dir = output_path / f"page_{idx}"
                    page_output_dir.mkdir(parents=True, exist_ok=True)

                    # 保存 JSON（结构化数据）
                    if hasattr(res, "save_to_json"):
                        res.save_to_json(save_path=str(page_output_dir))

                    # 保存 Markdown 文件（便于调试）
                    if hasattr(res, "save_to_markdown"):
                        res.save_to_markdown(save_path=str(page_output_dir))

                    # 收集结果用于合并
                    if hasattr(res, "markdown"):
                        markdown_list.append(res.markdown)
                        logger.info("   ✅ 提取成功")
                    
                    if hasattr(res, "json"):
                        json_list.append(res.json)

                except Exception as e:
                    logger.warning(f"   处理出错: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())

            # 使用官方方法合并所有页的 Markdown
            if hasattr(pipeline, "concatenate_markdown_pages"):
                markdown_text = pipeline.concatenate_markdown_pages(markdown_list)
                logger.info("   使用官方 concatenate_markdown_pages() 方法合并")
            else:
                # 降级方案：手动合并
                logger.warning("   未找到 concatenate_markdown_pages() 方法，使用降级方案")
                markdown_text = "\n\n---\n\n".join(
                    [str(md) if isinstance(md, str) else str(md.get("text", "")) for md in markdown_list]
                )

            # 保存合并后的 Markdown 文件
            markdown_file = output_path / "result.md"
            markdown_file.write_text(markdown_text, encoding="utf-8")
            logger.info(f"📄 Markdown 已保存: {markdown_file}")
            logger.info(f"   {len(result)} 页 | {len(markdown_text):,} 字符")

            # 始终保存 JSON 文件
            json_file = None
            if json_list:
                json_file = output_path / "result.json"
                # 合并所有页的 JSON
                combined_json = {"pages": json_list, "total_pages": len(result)}
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(combined_json, f, ensure_ascii=False, indent=2)
                logger.info(f"📄 JSON 已保存: {json_file}")
            else:
                logger.warning("⚠️  无法提取 JSON 数据")

            return {
                "success": True,
                "output_path": str(output_path),
                "markdown": markdown_text,
                "markdown_file": str(markdown_file),
                "json_file": str(json_file) if json_file else None,
                "result": result,
            }

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ OCR 解析失败:")
            logger.error(f"   错误类型: {type(e).__name__}")
            logger.error(f"   错误信息: {e}")
            logger.error("=" * 80)

            import traceback
            logger.debug("完整堆栈跟踪:")
            logger.debug(traceback.format_exc())

            raise

        finally:
            # 清理显存（无论成功或失败都执行）
            self.cleanup()


# 全局单例
_engine = None


def get_engine() -> PaddleOCRVLVLLMEngine:
    """获取全局引擎实例"""
    global _engine
    if _engine is None:
        _engine = PaddleOCRVLVLLMEngine()
    return _engine
