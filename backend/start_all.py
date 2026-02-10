#!/usr/bin/env python3
"""
MinerU Tianshu - 启动所有服务 (All-in-One)

1. VLLM Server (可选) - 端口 8003 (用于 PaddleOCR-VL)
2. API Server (FastAPI) - 端口 8000
3. LitServe Worker Pool - 端口 8001
4. Task Scheduler (可选) - 后台任务调度
5. MCP Server (可选) - 端口 8002

自动检查并下载 OCR 模型（PaddleOCR-VL）
支持 GPU 加速、任务队列、优先级管理
"""

import subprocess
import signal
import sys
import time
import os
import requests
from loguru import logger
from pathlib import Path
import argparse
from utils import parse_list_arg
from dotenv import load_dotenv


class TianshuLauncher:
    """天枢服务启动器"""

    def __init__(
        self,
        output_dir="/tmp/mineru_tianshu_output",
        api_port=8000,
        worker_port=8001,
        workers_per_device=1,
        devices="auto",
        accelerator="auto",
        enable_mcp=False,
        mcp_port=8002,
        # PaddleOCR VL VLLM 配置
        paddleocr_vl_vllm_engine_enabled=False,
        paddleocr_vl_vllm_api_list=[],
        # 本地 VLLM 启动配置
        start_local_vllm=False,
        vllm_model_path=None,
        vllm_port=8003,
        vllm_gpu_util=0.4,
        vllm_max_model_len=8192,
    ):
        self.output_dir = output_dir
        self.api_port = api_port
        self.worker_port = worker_port
        self.workers_per_device = workers_per_device
        self.devices = devices
        self.accelerator = accelerator
        self.enable_mcp = enable_mcp
        self.mcp_port = mcp_port
        self.processes = []
        
        # VLLM 相关配置
        self.paddleocr_vl_vllm_engine_enabled = paddleocr_vl_vllm_engine_enabled
        self.paddleocr_vl_vllm_api_list = paddleocr_vl_vllm_api_list
        self.start_local_vllm = start_local_vllm
        self.vllm_model_path = vllm_model_path
        self.vllm_port = vllm_port
        self.vllm_gpu_util = vllm_gpu_util
        self.vllm_max_model_len = vllm_max_model_len

    def check_ocr_models(self):
        """检查并下载所有 OCR 模型（异步，不阻塞启动）"""
        import threading

        # 1. 检查 PaddleOCR-VL 模型
        def check_paddleocr_vl():
            try:
                from paddleocr_vl import PaddleOCRVLEngine

                logger.info("🔍 Checking PaddleOCR-VL...")
                logger.info("   Note: PaddleOCR-VL models are auto-managed by PaddleOCR")
                
                # 简单初始化引擎（不触发下载）
                try:
                    PaddleOCRVLEngine()
                    logger.info("✅ PaddleOCR-VL engine initialized successfully")
                except Exception as e:
                    # 如果是因为缺少 API 连接导致的错误是正常的，只要包在就行
                    logger.debug(f"PaddleOCR-VL init check: {e}")

            except ImportError:
                logger.debug("PaddleOCR-VL not installed, skipping check")
            except Exception as e:
                logger.debug(f"PaddleOCR-VL check skipped: {e}")

        # 在后台线程中下载/检查模型
        thread_paddleocr = threading.Thread(target=check_paddleocr_vl, daemon=True)
        thread_paddleocr.start()

    def wait_for_vllm(self, port, timeout=300):
        """等待 VLLM 服务启动就绪"""
        start_time = time.time()
        health_url = f"http://localhost:{port}/v1/models"
        
        logger.info(f"⏳ Waiting for VLLM to load model at {health_url}...")
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url)
                if response.status_code == 200:
                    logger.info("✅ VLLM Service is ready!")
                    return True
            except requests.RequestException:
                pass
            
            # 检查进程是否还在
            for name, proc in self.processes:
                if name == "VLLM Service" and proc.poll() is not None:
                    logger.error("❌ VLLM process died while starting!")
                    return False
            
            time.sleep(2)
            
        logger.error("❌ Timeout waiting for VLLM to start.")
        return False

    def start_services(self):
        """启动所有服务"""
        logger.info("=" * 70)
        logger.info("🚀 MinerU Tianshu - AI Data Preprocessing Platform")
        logger.info("=" * 70)

        try:
            # 计算总服务数
            total_services = 3
            if self.enable_mcp: total_services += 1
            if self.start_local_vllm: total_services += 1
            
            current_step = 1

            # ---------------------------------------------------------
            # 0. (可选) 启动本地 VLLM 服务
            # ---------------------------------------------------------
            if self.start_local_vllm:
                logger.info(f"🧠 [{current_step}/{total_services}] Starting Local VLLM Service...")
                
                if not self.vllm_model_path:
                    logger.error("❌ --vllm-model-path is required when --start-local-vllm is enabled")
                    return False

                # 构建 VLLM 启动命令
                # 使用 python -m vllm.entrypoints.openai.api_server 以确保使用当前环境
                vllm_cmd = [
                    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", self.vllm_model_path,
                    "--port", str(self.vllm_port),
                    "--gpu-memory-utilization", str(self.vllm_gpu_util),
                    "--max-model-len", str(self.vllm_max_model_len),
                    "--trust-remote-code",
                    "--served-model-name", "paddleocr-vl" # 固定模型名称方便调用
                ]
                
                # 如果指定了 device，可能需要设置 CUDA_VISIBLE_DEVICES
                vllm_env = os.environ.copy()
                if self.devices != "auto" and isinstance(self.devices, list):
                     # 假设 VLLM 占用第一个设备，其余给 Worker，或者用户需自行通过环境变量控制
                     # 这里简单处理：让 VLLM 看所有卡，通过 tensor-parallel-size 控制（未在此处暴露）
                     pass

                vllm_proc = subprocess.Popen(vllm_cmd, env=vllm_env)
                self.processes.append(("VLLM Service", vllm_proc))
                
                # 等待 VLLM 就绪
                if not self.wait_for_vllm(self.vllm_port):
                    return False
                
                # 自动将本地 VLLM 地址加入列表
                local_vllm_url = f"http://localhost:{self.vllm_port}/v1"
                if local_vllm_url not in self.paddleocr_vl_vllm_api_list:
                    self.paddleocr_vl_vllm_api_list.append(local_vllm_url)
                    logger.info(f"🔗 Added local VLLM to API list: {local_vllm_url}")
                
                current_step += 1
                logger.info("")

            # ---------------------------------------------------------
            # 1. 启动 API Server
            # ---------------------------------------------------------
            logger.info(f"📡 [{current_step}/{total_services}] Starting API Server...")
            env = os.environ.copy()
            env["API_PORT"] = str(self.api_port)
            env["OUTPUT_PATH"] = self.output_dir
            api_proc = subprocess.Popen([sys.executable, "api_server.py"], cwd=Path(__file__).parent, env=env)
            self.processes.append(("API Server", api_proc))
            time.sleep(3)

            if api_proc.poll() is not None:
                logger.error("❌ API Server failed to start!")
                return False

            logger.info(f"   ✅ API Server started (PID: {api_proc.pid})")
            logger.info(f"   📖 API Docs: http://localhost:{self.api_port}/docs")
            current_step += 1
            logger.info("")

            # ---------------------------------------------------------
            # 2. 启动 LitServe Worker Pool
            # ---------------------------------------------------------
            logger.info(f"⚙️  [{current_step}/{total_services}] Starting LitServe Worker Pool...")
            worker_env = os.environ.copy()
            worker_env["WORKER_PORT"] = str(self.worker_port)
            worker_env["OUTPUT_PATH"] = self.output_dir

            worker_cmd = [
                sys.executable,
                "litserve_worker.py",
                "--output-dir", self.output_dir,
                "--accelerator", self.accelerator,
                "--workers-per-device", str(self.workers_per_device),
                "--port", str(self.worker_port),
                "--devices", str(self.devices) if isinstance(self.devices, str) else ",".join(map(str, self.devices)),
            ]

            # VLLM 参数透传
            if self.paddleocr_vl_vllm_engine_enabled:
                worker_cmd.extend(["--paddleocr-vl-vllm-engine-enabled"])
            
            # 此时 self.paddleocr_vl_vllm_api_list 可能已经包含本地启动的 VLLM
            worker_cmd.extend(["--paddleocr-vl-vllm-api-list", str(self.paddleocr_vl_vllm_api_list)])

            worker_proc = subprocess.Popen(worker_cmd, cwd=Path(__file__).parent, env=worker_env)
            self.processes.append(("LitServe Workers", worker_proc))
            time.sleep(5)

            if worker_proc.poll() is not None:
                logger.error("❌ LitServe Workers failed to start!")
                return False

            logger.info(f"   ✅ LitServe Workers started (PID: {worker_proc.pid})")
            current_step += 1
            logger.info("")

            # ---------------------------------------------------------
            # 3. 启动 Task Scheduler
            # ---------------------------------------------------------
            logger.info(f"🔄 [{current_step}/{total_services}] Starting Task Scheduler...")
            scheduler_cmd = [
                sys.executable,
                "task_scheduler.py",
                "--litserve-url", f"http://localhost:{self.worker_port}/predict",
                "--wait-for-workers",
            ]

            scheduler_proc = subprocess.Popen(scheduler_cmd, cwd=Path(__file__).parent)
            self.processes.append(("Task Scheduler", scheduler_proc))
            time.sleep(3)

            if scheduler_proc.poll() is not None:
                logger.error("❌ Task Scheduler failed to start!")
                return False

            logger.info(f"   ✅ Task Scheduler started (PID: {scheduler_proc.pid})")
            current_step += 1
            logger.info("")

            # ---------------------------------------------------------
            # 4. 启动 MCP Server（可选）
            # ---------------------------------------------------------
            if self.enable_mcp:
                logger.info(f"🔌 [{current_step}/{total_services}] Starting MCP Server...")
                mcp_env = os.environ.copy()
                mcp_env["API_BASE_URL"] = f"http://localhost:{self.api_port}"
                mcp_env["MCP_PORT"] = str(self.mcp_port)
                mcp_env["MCP_HOST"] = "0.0.0.0"

                mcp_proc = subprocess.Popen([sys.executable, "mcp_server.py"], cwd=Path(__file__).parent, env=mcp_env)
                self.processes.append(("MCP Server", mcp_proc))
                time.sleep(3)

                if mcp_proc.poll() is not None:
                    logger.error("❌ MCP Server failed to start!")
                    return False

                logger.info(f"   ✅ MCP Server started (PID: {mcp_proc.pid})")
                logger.info(f"   🌐 MCP Endpoint: http://localhost:{self.mcp_port}/mcp")
                logger.info("")

            # 启动成功
            logger.info("=" * 70)
            logger.info("✅ All Services Started Successfully!")
            logger.info("=" * 70)
            
            if self.start_local_vllm:
                logger.info(f"   • VLLM Service:       http://localhost:{self.vllm_port}/v1")
            
            logger.info(f"   • API Documentation:  http://localhost:{self.api_port}/docs")
            logger.info("")
            logger.info("⚠️  Press Ctrl+C to stop all services")
            
            self.check_ocr_models()
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start services: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.stop_services()
            return False

    def stop_services(self, signum=None, frame=None):
        """停止所有服务"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("⏹️  Stopping All Services...")
        logger.info("=" * 70)

        # 倒序关闭，先关后启动的，最后关最基础的服务
        for name, proc in reversed(self.processes):
            if proc.poll() is None:
                logger.info(f"   Stopping {name} (PID: {proc.pid})...")
                proc.terminate()

        # 等待进程结束
        for name, proc in reversed(self.processes):
            try:
                proc.wait(timeout=10)
                logger.info(f"   ✅ {name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"   ⚠️  {name} did not stop gracefully, forcing...")
                proc.kill()
                proc.wait()

        logger.info("=" * 70)
        logger.info("✅ All Services Stopped")
        logger.info("=" * 70)
        sys.exit(0)

    def wait(self):
        """等待所有服务"""
        try:
            while True:
                time.sleep(1)
                for name, proc in self.processes:
                    if proc.poll() is not None:
                        logger.error(f"❌ {name} unexpectedly stopped!")
                        self.stop_services()
                        return
        except KeyboardInterrupt:
            self.stop_services()


def main():
    """主函数"""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    
    parser = argparse.ArgumentParser(
        description="MinerU Tianshu - 统一启动脚本 (支持 VLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 基础配置
    parser.add_argument("--output-dir", type=str, default="/tmp/mineru_tianshu_output", help="输出目录")
    parser.add_argument("--api-port", type=int, default=8000, help="API端口")
    parser.add_argument("--worker-port", type=int, default=8001, help="Worker端口")
    
    # 硬件配置
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--devices", type=str, default="auto")
    
    # MCP 配置
    parser.add_argument("--enable-mcp", action="store_true", help="启用 MCP Server")
    parser.add_argument("--mcp-port", type=int, default=8002)
    
    # PaddleOCR VLLM 现有配置
    parser.add_argument("--paddleocr-vl-vllm-engine-enabled", action="store_true", default=False, help="启用 PaddleOCR VLLM 引擎逻辑")
    parser.add_argument("--paddleocr-vl-vllm-api-list", type=parse_list_arg, default=[], help="外部 VLLM API 列表")

    # [新增] 本地启动 VLLM 配置
    parser.add_argument("--start-local-vllm", action="store_true", help="是否在本地启动 VLLM 服务")
    parser.add_argument("--vllm-model-path", type=str, default=None, help="PaddleOCR-VL 模型路径 (当启用 local-vllm 时必填)")
    parser.add_argument("--vllm-port", type=int, default=8003, help="本地 VLLM 服务端口")
    parser.add_argument("--vllm-gpu-util", type=float, default=0.4, help="VLLM 显存占用比例 (0.0-1.0)")
    parser.add_argument("--vllm-max-model-len", type=int, default=8192, help="VLLM 最大上下文长度")

    args = parser.parse_args()

    # 处理 devices
    devices = args.devices
    if devices != "auto":
        try:
            devices = [int(d) for d in devices.split(",")]
        except ValueError:
            devices = "auto"

    # 逻辑校验：如果启动本地 VLLM，自动开启 engine enable
    if args.start_local_vllm:
        args.paddleocr_vl_vllm_engine_enabled = True
        logger.info("🚀 Local VLLM startup requested, auto-enabling PaddleOCR VLLM Engine.")

    if args.paddleocr_vl_vllm_engine_enabled:
        if not args.paddleocr_vl_vllm_api_list and not args.start_local_vllm:
             logger.error("启用 VLLM 引擎时，必须提供 --paddleocr-vl-vllm-api-list 或开启 --start-local-vllm")
             sys.exit(1)

    launcher = TianshuLauncher(
        output_dir=args.output_dir,
        api_port=args.api_port,
        worker_port=args.worker_port,
        workers_per_device=args.workers_per_device,
        devices=devices,
        accelerator=args.accelerator,
        enable_mcp=args.enable_mcp,
        mcp_port=args.mcp_port,
        # VLLM 参数
        paddleocr_vl_vllm_engine_enabled=args.paddleocr_vl_vllm_engine_enabled,
        paddleocr_vl_vllm_api_list=args.paddleocr_vl_vllm_api_list,
        start_local_vllm=args.start_local_vllm,
        vllm_model_path=args.vllm_model_path,
        vllm_port=args.vllm_port,
        vllm_gpu_util=args.vllm_gpu_util,
        vllm_max_model_len=args.vllm_max_model_len
    )

    signal.signal(signal.SIGINT, launcher.stop_services)
    signal.signal(signal.SIGTERM, launcher.stop_services)

    if launcher.start_services():
        launcher.wait()
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
