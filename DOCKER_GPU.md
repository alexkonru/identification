# Docker + GPU (CachyOS/Arch)

Если `./start_docker.sh` пишет:

`could not select device driver "" with capabilities: [[gpu]]`

это означает, что Docker не настроен на NVIDIA runtime.

## Что нужно на хосте

1. Установить `nvidia-container-toolkit`.
2. Прописать runtime для Docker (через `nvidia-ctk runtime configure --runtime=docker`).
3. Перезапустить Docker.
4. Проверить:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если команда выше показывает вашу видеокарту, проект можно запускать:

```bash
./start_docker.sh
```

## Переменные для тюнинга

- `VISION_CUDA_DEVICE_ID` — индекс GPU для vision (`""` = авто-перебор в приложении).
- `VISION_CUDA_MEM_LIMIT_MB` — лимит памяти CUDA EP для vision.
- `AUDIO_USE_CUDA` — `1` чтобы пробовать CUDA в audio, по умолчанию `0`.
- `AUDIO_CUDA_MEM_LIMIT_MB` — лимит памяти CUDA EP для audio.

## Клиент с хоста

Если запускаете `client/admin.py` на хосте, используйте:

```bash
GATEWAY_ADDR=127.0.0.1:50051 python client/admin.py
```

(порт проброшен из контейнера `gateway-service` в compose).

Клиент можно запускать и **без аргументов/переменных** — в коде уже default `127.0.0.1:50051`.

```bash
python client/admin.py
```

Если нужен другой адрес — тогда через переменную `GATEWAY_ADDR`.




Александр, классика! Мы споткнулись о специфический нейминг образов NVIDIA. Они не всегда создают «комбо-образы» (CUDA + cuDNN) для каждой минорной версии ОС сразу.

Кроме того, судя по логу `Sending build context to Docker daemon 838.1MB`, Docker всё ещё пытается затащить внутрь себя всю твою папку `target` с хоста. Это не только медленно, но и может вызвать конфликты.

Давай сделаем «хирургически точный» Dockerfile и исправим ошибки окружения.

### 1. Исправляем Docker-контекст (Важно!)

Создай в корне проекта файл `.dockerignore`, если его ещё нет. Это заставит Docker игнорировать мусор и старые сборки:

```text
# .dockerignore
target/
.git/
onnxruntime/
onnxruntime_sdk/
*.deb
*.tar.bz2

```

*После этого «Sending build context» должен упасть с 838 МБ до пары мегабайт.*

---

### 2. Устанавливаем Buildx (для CachyOS)

Чтобы не видеть ошибку `[WARN] buildx plugin не найден`, просто поставь его:

```bash
sudo pacman -S docker-buildx

```

---

### 3. Правильный Dockerfile для Ubuntu 24.04

Поскольку тег с `cudnn` не нашелся, мы возьмем чистый `devel` (где есть компиляторы) и поставим cuDNN вручную одной командой. В Ubuntu 24.04 теперь используется **cuDNN 9**, который отлично дружит с CUDA 12.4.

**Обнови свой Dockerfile:**

```dockerfile
# Берем базовый образ разработки
FROM nvidia/cuda:12.4.1-devel-ubuntu24.04

# Настройка окружения
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}

# Устанавливаем зависимости системы + cuDNN 9
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

# Ставим Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal --default-toolchain stable

WORKDIR /workspace/identification
COPY . .

# Сборка проекта
# Включаем динамическую линковку для ort, чтобы он подцепил системную куду
ENV ORT_STRATEGY=system
RUN cargo build --release

CMD ["./target/release/vision-worker"]

```

---

### Почему это должно сработать?

1. **Glibc:** В Ubuntu 24.04 версия glibc достаточно свежая (), чтобы удовлетворить требование `__isoc23_strtoll` (которое появилось в ).
2. **cuDNN 9:** Мы ставим `libcudnn9-dev-cuda-12` прямо из репозиториев NVIDIA, которые уже подключены в базовом образе.
3. **Чистота:** Благодаря `.dockerignore` мы не тащим скомпилированные под CachyOS файлы в Ubuntu.

### Что сделать сейчас:

1. Создай `.dockerignore`.
2. Обнови `Dockerfile`.
3. Запусти пересборку:
```bash
docker compose down
./start_docker.sh

```



Если на этапе `cargo build` возникнет ошибка, что `ort` не может найти библиотеку ONNX, дай знать — нам нужно будет просто уточнить путь в `LD_LIBRARY_PATH`. **Как думаешь, стоит нам заодно прописать пути к моделям в `docker-compose.yml`, чтобы не пересобирать контейнер при каждой смене весов?**