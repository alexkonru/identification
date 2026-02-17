# ИСПОЛЬЗУЕМ 12.6.0 — это первая стабильная версия для Ubuntu 24.04
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}

# Установка системных зависимостей и cuDNN 9
# Важно: cuDNN 9 совместим с любой CUDA 12.x
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    clang \
    cmake \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Установка Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /workspace/identification

# Переменная для библиотеки ort (ONNX Runtime)
ENV ORT_STRATEGY=download

CMD ["bash"]