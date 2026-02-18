# Универсальный базовый образ:
# - CUDA/cuDNN для GPU-ускорения,
# - полноценное Linux-окружение для CPU-фолбэка,
# - современная glibc (Ubuntu 24.04), чтобы избежать несовместимостей линковки.
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH} \
    # Для ort/ort-sys: скачивание официальных prebuilt-бинарников ONNX Runtime.
    ORT_STRATEGY=download

# Ставим инструменты сборки и runtime-зависимости:
# - libopenblas-dev: быстрый CPU backend,
# - libcudnn9-dev-cuda-12: заголовки/библиотеки cuDNN для CUDA-провайдера.
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
    libopenblas-dev \
    libcudnn9-dev-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

# Установка Rust toolchain.
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /workspace/identification

CMD ["bash"]
