#!/bin/bash
set -euo pipefail

# Запуск всей системы в Docker с поддержкой GPU.
# По умолчанию оба worker запускаются в CPU-режиме (безопасно для слабых GPU).
# Для GPU-режима выставьте перед запуском:
#   VISION_FORCE_CPU=0 AUDIO_FORCE_CPU=0 AUDIO_USE_CUDA=1 ./start_docker.sh

# Загружаем сохранённые runtime-настройки (если есть).
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-.server_runtime.env}"
if [ -f "$RUNTIME_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_FILE"
  set +a
fi

# Универсальная автонастройка CPU-параметров (не перетирает уже заданные значения).
detect_physical_cores() {
  local cores=""
  if command -v lscpu >/dev/null 2>&1; then
    cores="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l | tr -d ' ')"
  fi
  if [ -z "$cores" ] || [ "$cores" -le 0 ] 2>/dev/null; then
    local threads
    threads="$(nproc --all 2>/dev/null || echo 4)"
    cores=$(( threads / 2 ))
    [ "$cores" -le 0 ] && cores=1
  fi
  echo "$cores"
}

PHYSICAL_CORES="$(detect_physical_cores)"
VISION_AUTO_THREADS=$(( (PHYSICAL_CORES + 1) / 2 ))
AUDIO_AUTO_THREADS=$(( PHYSICAL_CORES - VISION_AUTO_THREADS ))
[ "$AUDIO_AUTO_THREADS" -le 0 ] && AUDIO_AUTO_THREADS=1

export VISION_INTRA_THREADS="${VISION_INTRA_THREADS:-$VISION_AUTO_THREADS}"
export AUDIO_INTRA_THREADS="${AUDIO_INTRA_THREADS:-$AUDIO_AUTO_THREADS}"
export VISION_INTER_THREADS="${VISION_INTER_THREADS:-1}"
export AUDIO_INTER_THREADS="${AUDIO_INTER_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

build_cpuset() {
  local start="$1"
  local count="$2"
  local end=$((start + count - 1))
  if [ "$count" -le 1 ]; then
    echo "$start"
  else
    echo "${start}-${end}"
  fi
}

VISION_CORE_COUNT=$(( (PHYSICAL_CORES + 1) / 2 ))
AUDIO_CORE_COUNT=$(( PHYSICAL_CORES - VISION_CORE_COUNT ))
if [ "$PHYSICAL_CORES" -le 1 ]; then
  VISION_CORE_COUNT=1
  AUDIO_CORE_COUNT=1
  export VISION_CPUSET="${VISION_CPUSET:-0}"
  export AUDIO_CPUSET="${AUDIO_CPUSET:-0}"
  export GATEWAY_CPUSET="${GATEWAY_CPUSET:-0}"
else
  [ "$AUDIO_CORE_COUNT" -le 0 ] && AUDIO_CORE_COUNT=1
  export VISION_CPUSET="${VISION_CPUSET:-$(build_cpuset 0 $VISION_CORE_COUNT)}"
  export AUDIO_CPUSET="${AUDIO_CPUSET:-$(build_cpuset $VISION_CORE_COUNT $AUDIO_CORE_COUNT)}"
  export GATEWAY_CPUSET="${GATEWAY_CPUSET:-0-$((PHYSICAL_CORES-1))}"
fi


# Определяем доступность GPU только если пользователь явно не зафиксировал режим.
if [ -z "${VISION_FORCE_CPU:-}" ]; then
  export VISION_FORCE_CPU=1
fi
if [ -z "${AUDIO_FORCE_CPU:-}" ]; then
  export AUDIO_FORCE_CPU=1
fi
if [ -z "${AUDIO_USE_CUDA:-}" ]; then
  export AUDIO_USE_CUDA=0
fi

# Перед запуском проверяем, что Docker видит compose-плагин.
docker compose version >/dev/null

# Проверяем, что Docker Engine умеет выдавать GPU контейнерам.
# Без nvidia-container-toolkit будет ошибка вида:
# "could not select device driver \"\" with capabilities: [[gpu]]"
if ! docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  cat <<'MSG'
[ERROR] Docker сейчас не может использовать GPU (NVIDIA runtime недоступен).

Что нужно сделать на хосте (CachyOS/Arch):
  1) Установить nvidia-container-toolkit.
  2) Настроить runtime для Docker и перезапустить docker.service.
  3) Проверить командой:
       docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi

После этого снова запусти:
  ./start_docker.sh
MSG
  exit 1
fi

# Собираем базовый образ один раз (без повторной сборки для каждого сервиса).
# Предпочитаем buildx (BuildKit), при отсутствии fallback на обычный docker build.
if docker buildx version >/dev/null 2>&1; then
  docker buildx build --load -t identification-rust-base:latest -f Dockerfile .
else
  echo "[WARN] buildx plugin не найден, используем legacy docker build." >&2
  docker build -t identification-rust-base:latest -f Dockerfile .
fi

# Запуск сервисов в фоне (без дополнительной сборки через compose).
docker compose up -d --no-build --force-recreate --remove-orphans db vision-worker audio-worker gateway-service

# Ждём, пока gateway откроет порт на хосте.
wait_gateway() {
  local retries=60
  for ((i=1; i<=retries; i++)); do
    if (echo > /dev/tcp/127.0.0.1/50051) >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

if ! wait_gateway; then
  echo "[ERROR] Gateway не поднялся на 127.0.0.1:50051 за отведённое время."
  echo "Текущий статус контейнеров:"
  docker compose ps || true
  echo
  echo "Последние логи gateway-service:"
  docker compose logs --tail=120 gateway-service || true
  exit 1
fi

# Печатаем состояние контейнеров.
docker compose ps

echo
echo "Сервисы запущены в Docker."
echo "Логи vision:  docker compose logs -f vision-worker"
echo "Логи audio:   docker compose logs -f audio-worker"
echo "Логи gateway: docker compose logs -f gateway-service"
echo
echo "Проверка GPU внутри vision-контейнера:"
echo "  docker compose exec vision-worker nvidia-smi"
echo
echo "Для клиента с хоста используй gateway: 127.0.0.1:50051"

echo
echo "Остановка Docker-сервисов:"
echo "  docker compose stop      # остановить контейнеры"
echo "  docker compose down      # остановить и удалить контейнеры/сеть"
