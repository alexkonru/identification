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
