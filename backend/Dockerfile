# ============================================================================
# Stage 1: Base Image with CUDA Support (CUDA 13.0)
# ============================================================================
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS base

# 配置镜像源和 APT 优化
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list.d/ubuntu.sources

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
    python3.12 python3.12-dev python3-pip python3.12-venv \
    git wget curl libgomp1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1 libgstreamer1.0-0 ffmpeg antiword pandoc \
    libreoffice libreoffice-writer libreoffice-calc libreoffice-impress \
    fonts-noto-cjk fonts-wqy-zenhei fonts-wqy-microhei fontconfig \
    && fc-cache -fv && rm -rf /var/lib/apt/lists/*

# 符号链接
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# [关键修复] 立即升级构建工具，修复 Python 3.12 的 'ImpImporter' 兼容性报错
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade --ignore-installed pip "setuptools>=70.0.0" wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp
COPY backend/requirements.txt /tmp/requirements.txt

# Step 1: 预装 NumPy 和 OpenCV Headless (防止后续冲突)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "numpy==1.26.4" "opencv-python-headless>=4.13.0.90" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Step 2: 安装 vLLM (cu130)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "vllm>=0.15.1" --extra-index-url https://download.pytorch.org/whl/cu130 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Step 3: 安装预编译的 FlashAttention (cu130 + torch2.10)
# [关键更新] 使用用户指定的 Torch 2.10 兼容版本
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-linux_x86_64.whl \
    --break-system-packages

# Step 4: 安装 PaddlePaddle GPU (cu130)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
    --break-system-packages

# Step 5: 安装核心依赖 (采用 --no-deps 策略防止污染环境)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "mineru[core]>=2.7.6" "albumentations>=1.4.11" "albucore>=0.0.15" \
    --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Step 6: 自动清理 requirements.txt 并安装剩余依赖
# 移除所有可能导致版本降级的包
RUN --mount=type=cache,target=/root/.cache/pip \
    sed -i '/torch/d; /nvidia/d; /opencv/d; /numpy/d; /albumentations/d; /paddle/d' /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --break-system-packages

# Step 7: [最终加固] 强制重新对齐 Torch 全家桶到 cu130 (Torch 2.10)
# 这一步必须与 Step 3 的 FlashAttention 版本匹配
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --force-reinstall \
    "torch==2.10.0+cu130" \
    "torchvision>=0.22.0+cu130" \
    "torchaudio==2.10.0+cu130" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --break-system-packages

# 验证安装
RUN python -c "import torch; import torchvision; print(f'Final PyTorch: {torch.__version__}, Torchvision: {torchvision.__version__}')" && \
    pip show paddlepaddle-gpu | grep Version

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application
WORKDIR /app
COPY backend/ /app/backend/
COPY pyproject.toml /app/
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
COPY scripts/init-models.sh /usr/local/bin/init-models.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh /usr/local/bin/init-models.sh

RUN mkdir -p /app/models /app/data/uploads /app/data/output /app/logs
WORKDIR /app/backend

EXPOSE 8000 8001 8002
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["api", "python", "api_server.py"]
