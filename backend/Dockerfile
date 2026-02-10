# ============================================================================
# Stage 1: Base Image with CUDA Support
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
# [修复] 添加 --ignore-installed 参数
# 原因: Ubuntu 24.04 系统自带的 pip 是 apt 安装的，没有 RECORD 文件，无法被 pip 卸载。
# --ignore-installed 会强制覆盖安装新版本，而不尝试卸载旧版本。
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
# 核心依赖安装顺序优化
# 1. vLLM (自动拉取匹配的 PyTorch GPU 版)
# 2. PaddlePaddle (与 PyTorch 共存)
# 3. MinerU & 应用依赖
# ----------------------------------------------------------------------------

# Step 1: 安装 vLLM (让 vLLM 自动决定 PyTorch 版本)
# [关键] 必须指定 extra-index-url 为 CUDA 源，否则会自动安装 CPU 版 PyTorch！
# [关键] 使用 --no-cache-dir 避免缓存了错误的 CPU 版包
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing vLLM and letting it pull PyTorch..." && \
    pip install "vllm>=0.15.1" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=600 \
    || echo "Warning: vLLM installation faced issues, proceeding to check..."

# Step 2: 安装 PaddlePaddle GPU
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing PaddlePaddle..." && \
    pip install paddlepaddle-gpu==3.3.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages \
    --default-timeout=600

# Step 3: 预装关键 Python 3.12 依赖 (构建隔离)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "kiwisolver>=1.4.5" "Pillow>=11.0.0" \
    "numpy>=1.26.0,<2.0.0" "setuptools>=75.0.0" "lxml>=5.3.0" \
    "packaging" "ninja" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 4: 安装 MinerU 核心 (使用 --no-deps 防止它破坏 vLLM 的 PyTorch)
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing MinerU Core..." && \
    pip install "mineru[core]>=2.7.6" --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 5: 安装其他核心包 (已修复 Tokenizers 冲突)
# 变更点：
# 1. 显式允许 tokenizers>=0.22.0 以兼容 transformers 4.57.6
# 2. transformers 使用 ==4.57.6 保持确定性
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

# Step 6: 确保 Albumentations 版本兼容性
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install 'albumentations>=1.3.1,<1.4.0' 'albucore>=0.0.13,<0.0.17' \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages

# Step 7: 安装剩余依赖
RUN --mount=type=cache,target=/root/.cache/pip \
    echo "Installing remaining requirements..." && \
    pip install -r /tmp/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=600

# 验证安装
RUN python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')" && \
    python -c "import paddle; print(f'Paddle: {paddle.__version__}')"

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
