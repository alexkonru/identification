#!/bin/bash
set -euo pipefail

BASE_IMAGE="identification-rust-base:latest"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-.server_runtime.env}"
export ORT_STRATEGY="${ORT_STRATEGY:-download}"

echo "--- [1/4] Подготовка окружения ---"
if [ -f "$RUNTIME_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_FILE"
  set +a
  echo "[INFO] Загружены настройки из $RUNTIME_ENV_FILE"
fi

# Авто-настройка CPU потоков/пиннинга
if command -v nproc >/dev/null 2>&1; then
  CPU_THREADS="$(nproc --all)"
else
  CPU_THREADS=4
fi
PHYSICAL_CORES=$(( CPU_THREADS / 2 ))
[ "$PHYSICAL_CORES" -le 0 ] && PHYSICAL_CORES=1
VISION_AUTO_THREADS=$(( (PHYSICAL_CORES + 1) / 2 ))
AUDIO_AUTO_THREADS=$(( PHYSICAL_CORES - VISION_AUTO_THREADS ))
[ "$AUDIO_AUTO_THREADS" -le 0 ] && AUDIO_AUTO_THREADS=1

export VISION_INTRA_THREADS="${VISION_INTRA_THREADS:-$VISION_AUTO_THREADS}"
export AUDIO_INTRA_THREADS="${AUDIO_INTRA_THREADS:-$AUDIO_AUTO_THREADS}"
export VISION_INTER_THREADS="${VISION_INTER_THREADS:-1}"
export AUDIO_INTER_THREADS="${AUDIO_INTER_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VISION_CPUSET="${VISION_CPUSET:-0-$((VISION_AUTO_THREADS-1))}"
export AUDIO_CPUSET="${AUDIO_CPUSET:-$VISION_AUTO_THREADS-$((PHYSICAL_CORES-1))}"
export GATEWAY_CPUSET="${GATEWAY_CPUSET:-0-$((PHYSICAL_CORES-1))}"
export DOOR_AGENT_CPUSET="${DOOR_AGENT_CPUSET:-0-1}"

GPU_ENABLED=false
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  GPU_ENABLED=true
  echo "[INFO] NVIDIA GPU обнаружена и доступна в Docker."
else
  echo "[WARN] GPU не найдена/не проброшена. Переключаем worker'ы в CPU режим."
  export VISION_FORCE_CPU="${VISION_FORCE_CPU:-1}"
  export AUDIO_FORCE_CPU="${AUDIO_FORCE_CPU:-1}"
  export AUDIO_USE_CUDA="${AUDIO_USE_CUDA:-0}"
fi

echo "--- [2/4] Сборка базового образа ---"
if docker buildx version >/dev/null 2>&1; then
  docker buildx build --load -t "$BASE_IMAGE" -f Dockerfile .
else
  docker build -t "$BASE_IMAGE" -f Dockerfile .
fi

echo "--- [3/4] Предварительная сборка workspace ---"
docker run --rm \
  -v "$(pwd)":/workspace/identification \
  -v cargo-registry:/usr/local/cargo/registry \
  -v cargo-git:/usr/local/cargo/git \
  -e ORT_STRATEGY="$ORT_STRATEGY" \
  -w /workspace/identification \
  "$BASE_IMAGE" \
  cargo build --release --workspace

echo "--- [4/4] Запуск сервисов через Docker Compose ---"
docker compose up -d --no-build --force-recreate --remove-orphans db vision-worker audio-worker gateway-service door-agent

wait_gateway() {
  echo -n "[INFO] Ожидание запуска Gateway (127.0.0.1:50051)..."
  local retries=30
  while ! (echo > /dev/tcp/127.0.0.1/50051) >/dev/null 2>&1; do
    sleep 2
    retries=$((retries - 1))
    echo -n "."
    if [ "$retries" -le 0 ]; then
      echo
      echo "[ERROR] Gateway не ответил за 60 секунд."
      return 1
    fi
  done
  echo " Готово!"
}

if wait_gateway; then
  echo "----------------------------------------------------"
  echo "Система успешно запущена!"
  echo "Gateway: http://127.0.0.1:50051"
  echo "Door-agent: http://127.0.0.1:50054"
  echo "Для мониторинга используй: docker compose logs -f"
  echo "----------------------------------------------------"
else
  echo "[CRITICAL] Ошибка при запуске. Проверь логи: docker compose logs gateway-service door-agent"
  exit 1
fi
