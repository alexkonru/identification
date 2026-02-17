# Docker + GPU/CPU (CachyOS/Arch)

Эта конфигурация сделана так, чтобы:
- **стабильно запускаться на CPU** по умолчанию (для MX230/Vega и слабых GPU),
- и включать **GPU-режим** одной командой на более мощной машине.

## 1) Проверка GPU-runtime Docker

Если `./start_docker.sh` пишет:

`could not select device driver "" with capabilities: [[gpu]]`

значит Docker не настроен на NVIDIA runtime.

Проверь:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Если команда не работает — нужно установить/настроить `nvidia-container-toolkit`.

## 2) Режимы запуска

### Безопасный (по умолчанию, CPU)

```bash
./start_docker.sh
```

По умолчанию в `docker-compose.yml`:
- `VISION_FORCE_CPU=1`
- `AUDIO_FORCE_CPU=1`
- `AUDIO_USE_CUDA=0`

### GPU-режим (явно)

```bash
VISION_FORCE_CPU=0 AUDIO_FORCE_CPU=0 AUDIO_USE_CUDA=1 ./start_docker.sh
```

При необходимости ограничь память CUDA EP:

```bash
VISION_CUDA_MEM_LIMIT_MB=1024 AUDIO_CUDA_MEM_LIMIT_MB=256 \
VISION_FORCE_CPU=0 AUDIO_FORCE_CPU=0 AUDIO_USE_CUDA=1 ./start_docker.sh
```

## 3) Остановка

```bash
./stop_docker.sh
# или с удалением томов (ОСТОРОЖНО, удалит данные БД):
./stop_docker.sh --volumes
```

## 4) Клиент с хоста

```bash
python client/admin.py
```

По умолчанию клиент уже использует `127.0.0.1:50051`.
