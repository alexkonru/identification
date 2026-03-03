#!/bin/bash
set -euo pipefail

BASE_IMAGE="identification-rust-base:latest"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-.server_runtime.env}"
export ORT_STRATEGY="${ORT_STRATEGY:-download}"

detect_physical_cores() {
  local cores_per_socket sockets phys
  if command -v lscpu >/dev/null 2>&1; then
    cores_per_socket="$(lscpu | awk -F: '/^Core\(s\) per socket:/{gsub(/ /,"",$2); print $2; exit}')"
    sockets="$(lscpu | awk -F: '/^Socket\(s\):/{gsub(/ /,"",$2); print $2; exit}')"
    if [[ "${cores_per_socket:-}" =~ ^[0-9]+$ ]] && [[ "${sockets:-}" =~ ^[0-9]+$ ]] && [ "$cores_per_socket" -gt 0 ] && [ "$sockets" -gt 0 ]; then
      phys=$((cores_per_socket * sockets))
      echo "$phys"
      return 0
    fi
  fi
  if [ -r /proc/cpuinfo ]; then
    phys="$(awk '
      /^physical id/ {p=$4}
      /^core id/ {c=$4; if (p != "" && c != "") {k=p":"c; if (!seen[k]++) n++}}
      END {if (n > 0) print n}
    ' /proc/cpuinfo)"
    if [[ "${phys:-}" =~ ^[0-9]+$ ]] && [ "$phys" -gt 0 ]; then
      echo "$phys"
      return 0
    fi
  fi
  if command -v nproc >/dev/null 2>&1; then
    local threads
    threads="$(nproc --all)"
    if [ -r /sys/devices/system/cpu/smt/active ] && [ "$(cat /sys/devices/system/cpu/smt/active 2>/dev/null)" = "1" ]; then
      echo $((threads / 2))
    else
      echo "$threads"
    fi
    return 0
  fi
  echo 2
}

runtime_threads_plan() {
  local phys="$1" vision audio reserve workers
  [ "$phys" -lt 1 ] && phys=1
  reserve=2
  [ "$phys" -le 4 ] && reserve=1
  workers=$((phys - reserve))
  [ "$workers" -lt 2 ] && workers=2
  # Перекос в сторону vision (примерно 65/35), но оба worker должны оставаться активными.
  vision=$(( (workers * 2 + 1) / 3 ))
  [ "$vision" -lt 1 ] && vision=1
  audio=$((workers - vision))
  [ "$audio" -lt 1 ] && audio=1
  echo "$vision $audio"
}

build_cpuset_ranges() {
  local logical="$1" phys="$2" reserve="$3" vision="$4" audio="$5"
  [ "$logical" -lt 1 ] && logical=1
  [ "$phys" -lt 1 ] && phys=1
  local worker_phys=$((phys - reserve))
  [ "$worker_phys" -lt 2 ] && worker_phys=2
  local worker_logical=$(( logical * worker_phys / phys ))
  [ "$worker_logical" -lt 2 ] && worker_logical=2
  [ "$worker_logical" -gt "$logical" ] && worker_logical="$logical"
  local vision_logical=$(( worker_logical * vision / (vision + audio) ))
  [ "$vision_logical" -lt 1 ] && vision_logical=1
  [ "$vision_logical" -ge "$worker_logical" ] && vision_logical=$((worker_logical - 1))
  [ "$vision_logical" -lt 1 ] && vision_logical=1
  local audio_start="$vision_logical"
  local audio_end=$((worker_logical - 1))
  local vision_end=$((vision_logical - 1))
  [ "$vision_end" -lt 0 ] && vision_end=0
  [ "$audio_start" -gt "$audio_end" ] && audio_start="$audio_end"
  [ "$audio_start" -lt 0 ] && audio_start=0
  [ "$audio_end" -lt 0 ] && audio_end=0
  echo "0-$vision_end $audio_start-$audio_end 0-$((logical - 1)) 0-$(( (logical > 1) ? 1 : 0 ))"
}

echo "--- [1/4] Подготовка окружения ---"
if [ -f "$RUNTIME_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_FILE"
  set +a
  echo "[INFO] Загружены настройки из $RUNTIME_ENV_FILE"
fi

# Автонастройка числа потоков и закрепления по CPU
if command -v nproc >/dev/null 2>&1; then
  CPU_THREADS="$(nproc --all)"
else
  CPU_THREADS=4
fi
PHYSICAL_CORES="$(detect_physical_cores)"
[ "$PHYSICAL_CORES" -le 0 ] && PHYSICAL_CORES=1
read -r VISION_AUTO_THREADS AUDIO_AUTO_THREADS < <(runtime_threads_plan "$PHYSICAL_CORES")
RESERVE_SYSTEM=2
[ "$PHYSICAL_CORES" -le 4 ] && RESERVE_SYSTEM=1
read -r VISION_CPUSET_AUTO AUDIO_CPUSET_AUTO GATEWAY_CPUSET_AUTO DOOR_CPUSET_AUTO < <(build_cpuset_ranges "$CPU_THREADS" "$PHYSICAL_CORES" "$RESERVE_SYSTEM" "$VISION_AUTO_THREADS" "$AUDIO_AUTO_THREADS")

export VISION_INTRA_THREADS="${VISION_INTRA_THREADS:-$VISION_AUTO_THREADS}"
export AUDIO_INTRA_THREADS="${AUDIO_INTRA_THREADS:-$AUDIO_AUTO_THREADS}"
export VISION_INTER_THREADS="${VISION_INTER_THREADS:-1}"
export AUDIO_INTER_THREADS="${AUDIO_INTER_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VISION_CPUSET="${VISION_CPUSET:-$VISION_CPUSET_AUTO}"
export AUDIO_CPUSET="${AUDIO_CPUSET:-$AUDIO_CPUSET_AUTO}"
export GATEWAY_CPUSET="${GATEWAY_CPUSET:-$GATEWAY_CPUSET_AUTO}"
export DOOR_AGENT_CPUSET="${DOOR_AGENT_CPUSET:-$DOOR_CPUSET_AUTO}"
export DOOR_BACKGROUND_ENABLED="${DOOR_BACKGROUND_ENABLED:-1}"
export DOOR_BG_FRAMES="${DOOR_BG_FRAMES:-8}"
export DOOR_MAX_CLIP_FRAMES="${DOOR_MAX_CLIP_FRAMES:-8}"
export CAMERA_LOCK_DIR="${CAMERA_LOCK_DIR:-/workspace/identification/locks}"
export PIPELINE_AUTO_START="${PIPELINE_AUTO_START:-0}"

echo "[INFO] CPU: logical=$CPU_THREADS, physical=$PHYSICAL_CORES"
echo "[INFO] Runtime threads: VISION_INTRA_THREADS=$VISION_INTRA_THREADS, AUDIO_INTRA_THREADS=$AUDIO_INTRA_THREADS"
echo "[INFO] CPU pinning: VISION_CPUSET=$VISION_CPUSET, AUDIO_CPUSET=$AUDIO_CPUSET, GATEWAY_CPUSET=$GATEWAY_CPUSET, DOOR_AGENT_CPUSET=$DOOR_AGENT_CPUSET"

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
  if [ "$PIPELINE_AUTO_START" = "1" ]; then
    docker compose exec -T gateway-service bash -lc \
      "printf '1\n' > \"\${SYSTEM_RUN_FLAG_PATH:-/workspace/identification/.system_run}\""
    echo "[INFO] Флаг pipeline установлен в 1 (автозапуск включен)."
  else
    docker compose exec -T gateway-service bash -lc \
      "printf '0\n' > \"\${SYSTEM_RUN_FLAG_PATH:-/workspace/identification/.system_run}\""
    echo "[INFO] Флаг pipeline установлен в 0 (ожидание ручного запуска из клиента)."
  fi
  echo "----------------------------------------------------"
  echo "Система успешно запущена!"
  echo "Gateway: http://127.0.0.1:50051"
  echo "Door-agent: запущен во внутренней сети Docker"
  echo "Для мониторинга используй: docker compose logs -f"
  echo "----------------------------------------------------"
else
  echo "[CRITICAL] Ошибка при запуске. Проверь логи: docker compose logs gateway-service door-agent"
  exit 1
fi
