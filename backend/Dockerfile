# ============================================================================
# Stage 1: Base Image with CUDA Support (CUDA 13.0)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 配置镜像源和 APT 优化
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources

RUN echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::Retries "5";' >> /etc/apt/apt.conf.d/99custom && \
    echo 'Acquire::http::Timeout "120";' >> /etc/apt/apt.conf.d/99custom

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PIP_CACHE_DIR=/root/.cache/pip

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    python3.12-venv \
    git \
    wget \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    libgstreamer1.0-0 \
    ffmpeg \
    antiword \
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# 安装 LibreOffice (固定大版本 24.8.x)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libreoffice \
    libreoffice-writer \
    libreoffice-calc \
    libreoffice-impress \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    fontconfig \
    && fc-cache -fv \
    && rm -rf /var/lib/apt/lists/*

# 符号链接
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# 升级 pip
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --ignore-installed pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=300

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# ----------------------------------------------------------------------------
# 核心依赖安装顺序
# ----------------------------------------------------------------------------

# Step 1: 安装 vLLM (最关键依赖，由此确定 PyTorch 版本为 cu130)
# [关键] 指定 --extra-index-url 为 cu130，确保拉取正确的 PyTorch
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM and letting it pull PyTorch (cu130)..." && \
    pip install "vllm>=0.15.1" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=600 \
    || echo "Warning: vLLM installation faced issues, proceeding to check..."

# Step 2: 预装构建工具
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "kiwisolver>=1.4.5" "Pillow>=11.0.0" \
    "numpy>=1.26.0,<2.0.0" "setuptools>=75.0.0" "lxml>=5.3.0" \
    "packaging" "ninja" "psutil" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 3: 安装预编译的 FlashAttention Wheel
# [关键] 使用匹配 CUDA 13.0 + Torch 2.9 的版本
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing FlashAttention Wheel..." && \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl \
    --break-system-packages

# Step 4: 安装 PaddlePaddle GPU
# [关键] 指定 paddlepaddle 仓库的 cu130 目录
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages \
    --default-timeout=600

# Step 5: 安装 MinerU 核心 (使用 --no-deps 防止它破坏 vLLM 的 PyTorch)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU Core..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 6: 安装其他核心包
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing other core packages..." && \
    pip install paddleocr[doc-parser] \
    transformers==4.57.6 tokenizers>=0.22.0 \
    fastapi uvicorn litserve aiohttp \
    PyMuPDF Pillow img2pdf einops easydict addict loguru modelscope \
    minio markitdown \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=300 \
    --retries 5

# Step 7: 解决 OpenCV 版本冲突
# [关键] vLLM 需要 opencv-python-headless>=4.13，显式安装以满足需求
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy<2.0.0" "albumentations>=1.4.11" "opencv-python-headless>=4.13.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 8: 安装剩余依赖 (剔除 torch 相关)
# [关键修正] 从 requirements.txt 中剔除 torch, torchvision, torchaudio
# 防止 pip install -r 再次拉取不兼容的 PyPI 版本 (如 cu128 或 cpu)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Processing requirements.txt and installing..." && \
    # 使用 sed 删除相关行
    sed -i '/^torch/d' /tmp/requirements.txt && \
    sed -i '/^torchvision/d' /tmp/requirements.txt && \
    sed -i '/^torchaudio/d' /tmp/requirements.txt && \
    # 打印处理后的文件确认
    cat /tmp/requirements.txt | grep -E "torch|vision|audio" || echo "Torch related packages removed from requirements.txt" && \
    pip install -r /tmp/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=600

# 验证安装
RUN python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')" && \
    echo "Verifying PaddlePaddle installation..." && \
    pip show paddlepaddle-gpu | grep Version && \
    echo "PaddlePaddle installed"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

WORKDIR /app

# 复制后端代码
COPY backend/ /app/backend/
COPY pyproject.toml /app/

# 复制启动脚本
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/init-models.sh /usr/local/bin/init-models.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/init-models.sh

RUN mkdir -p /app/models /app/data /app/uploads /app/output /app/logs
WORKDIR /app/backend

EXPOSE 8000 8001 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]
