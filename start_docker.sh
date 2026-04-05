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
  local capacity="$1" vision audio reserve workers
  [ "$capacity" -lt 1 ] && capacity=1
  reserve=2
  [ "$capacity" -le 8 ] && reserve=1
  workers=$((capacity - reserve))
  [ "$workers" -lt 2 ] && workers=2
  # Перекос в сторону vision (примерно 75/25), но оба worker должны оставаться активными.
  vision=$(( (workers * 3 + 1) / 4 ))
  [ "$vision" -lt 1 ] && vision=1
  audio=$((workers - vision))
  [ "$audio" -lt 1 ] && audio=1
  echo "$vision $audio"
}

physical_core_groups() {
  if command -v lscpu >/dev/null 2>&1; then
    local out
    out="$(lscpu -p=CPU,CORE,SOCKET 2>/dev/null | awk -F, '
      !/^#/ {
        key=$3 ":" $2
        if (!(key in seen)) {
          seen[key]=++n
          order[n]=key
        }
        if (groups[key] == "") {
          groups[key]=$1
        } else {
          groups[key]=groups[key] "," $1
        }
      }
      END {
        for (i=1; i<=n; i++) {
          print groups[order[i]]
        }
      }
    ')"
    if [ -n "${out:-}" ]; then
      printf '%s\n' "$out"
      return 0
    fi
  fi
  return 1
}

join_groups_csv() {
  local start="$1" count="$2"
  shift 2
  local groups=("$@")
  local result="" i end
  end=$((start + count))
  [ "$start" -lt 0 ] && start=0
  [ "$count" -lt 1 ] && count=1
  for ((i=start; i<end && i<${#groups[@]}; i++)); do
    if [ -z "$result" ]; then
      result="${groups[$i]}"
    else
      result="$result,${groups[$i]}"
    fi
  done
  printf '%s\n' "$result"
}

count_cpuset_cpus() {
  local spec="$1" total=0 part start end
  [ -z "${spec:-}" ] && { echo 1; return 0; }
  IFS=',' read -r -a parts <<< "$spec"
  for part in "${parts[@]}"; do
    if [[ "$part" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${part%-*}"
      end="${part#*-}"
      if [ "$end" -ge "$start" ]; then
        total=$((total + end - start + 1))
      fi
    elif [[ "$part" =~ ^[0-9]+$ ]]; then
      total=$((total + 1))
    fi
  done
  [ "$total" -lt 1 ] && total=1
  echo "$total"
}

build_cpuset_ranges() {
  local logical="$1" phys="$2" reserve="$3" vision="$4" audio="$5"
  local groups total_groups worker_phys reserve_groups vision_set audio_set gateway_set door_set system_set
  if mapfile -t groups < <(physical_core_groups); then
    total_groups="${#groups[@]}"
    if [ "$total_groups" -gt 0 ]; then
      worker_phys=$((total_groups - reserve))
      [ "$worker_phys" -lt 2 ] && worker_phys=2
      [ "$worker_phys" -gt "$total_groups" ] && worker_phys="$total_groups"
      reserve_groups=$((total_groups - worker_phys))
      [ "$reserve_groups" -lt 1 ] && reserve_groups=1
      [ "$vision" -gt "$worker_phys" ] && vision="$worker_phys"
      [ "$vision" -lt 1 ] && vision=1
      audio=$((worker_phys - vision))
      [ "$audio" -lt 1 ] && audio=1
      if [ $((vision + audio)) -gt "$worker_phys" ]; then
        audio=$((worker_phys - vision))
        [ "$audio" -lt 1 ] && audio=1
      fi
      vision_set="$(join_groups_csv 0 "$vision" "${groups[@]}")"
      audio_set="$(join_groups_csv "$vision" "$audio" "${groups[@]}")"
      system_set="$(join_groups_csv "$worker_phys" "$reserve_groups" "${groups[@]}")"
      [ -z "$system_set" ] && system_set="$(join_groups_csv 0 1 "${groups[@]}")"
      gateway_set="$system_set"
      door_set="$system_set"
      echo "$vision_set $audio_set $gateway_set $door_set"
      return 0
    fi
  fi

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

export RUNTIME_MODE="${RUNTIME_MODE:-auto}"

# Автонастройка числа потоков и закрепления по CPU
if command -v nproc >/dev/null 2>&1; then
  CPU_THREADS="$(nproc --all)"
else
  CPU_THREADS=4
fi
PHYSICAL_CORES="$(detect_physical_cores)"
[ "$PHYSICAL_CORES" -le 0 ] && PHYSICAL_CORES=1
export HOST_CPU_THREADS="$CPU_THREADS"
export HOST_PHYSICAL_CORES="$PHYSICAL_CORES"
THREAD_PLAN_BUDGET="$PHYSICAL_CORES"
[ "${RUNTIME_MODE:-auto}" = "cpu" ] && THREAD_PLAN_BUDGET="$CPU_THREADS"
read -r VISION_AUTO_THREADS AUDIO_AUTO_THREADS < <(runtime_threads_plan "$THREAD_PLAN_BUDGET")
RESERVE_SYSTEM=2
[ "$PHYSICAL_CORES" -le 8 ] && RESERVE_SYSTEM=1
read -r VISION_CPUSET_AUTO AUDIO_CPUSET_AUTO GATEWAY_CPUSET_AUTO DOOR_CPUSET_AUTO < <(build_cpuset_ranges "$CPU_THREADS" "$PHYSICAL_CORES" "$RESERVE_SYSTEM" "$VISION_AUTO_THREADS" "$AUDIO_AUTO_THREADS")

if [ "$RUNTIME_MODE" = "auto" ]; then
  export VISION_INTRA_THREADS="$VISION_AUTO_THREADS"
  export AUDIO_INTRA_THREADS="$AUDIO_AUTO_THREADS"
  export VISION_INTER_THREADS=1
  export AUDIO_INTER_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VISION_CPUSET="$VISION_CPUSET_AUTO"
  export AUDIO_CPUSET="$AUDIO_CPUSET_AUTO"
  export GATEWAY_CPUSET="$GATEWAY_CPUSET_AUTO"
  export DOOR_AGENT_CPUSET="$DOOR_CPUSET_AUTO"
elif [ "$RUNTIME_MODE" = "cpu" ]; then
  export VISION_INTRA_THREADS="$VISION_AUTO_THREADS"
  export AUDIO_INTRA_THREADS="$AUDIO_AUTO_THREADS"
  export VISION_INTER_THREADS=1
  export AUDIO_INTER_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export VISION_CPUSET="$VISION_CPUSET_AUTO"
  export AUDIO_CPUSET="$AUDIO_CPUSET_AUTO"
  export GATEWAY_CPUSET="$GATEWAY_CPUSET_AUTO"
  export DOOR_AGENT_CPUSET="$DOOR_CPUSET_AUTO"
else
  export VISION_INTRA_THREADS="${VISION_INTRA_THREADS:-$VISION_AUTO_THREADS}"
  export AUDIO_INTRA_THREADS="${AUDIO_INTRA_THREADS:-$AUDIO_AUTO_THREADS}"
  export VISION_INTER_THREADS="${VISION_INTER_THREADS:-1}"
  export AUDIO_INTER_THREADS="${AUDIO_INTER_THREADS:-1}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
  export VISION_CPUSET="${VISION_CPUSET:-$VISION_CPUSET_AUTO}"
  export AUDIO_CPUSET="${AUDIO_CPUSET:-$AUDIO_CPUSET_AUTO}"
  export GATEWAY_CPUSET="${GATEWAY_CPUSET:-$GATEWAY_CPUSET_AUTO}"
  export DOOR_AGENT_CPUSET="${DOOR_AGENT_CPUSET:-$DOOR_CPUSET_AUTO}"
fi
export DOOR_BACKGROUND_ENABLED="${DOOR_BACKGROUND_ENABLED:-1}"
export DOOR_BG_FRAMES="${DOOR_BG_FRAMES:-4}"
export DOOR_BG_TICK_MS="${DOOR_BG_TICK_MS:-100}"
export DOOR_BG_COOLDOWN_MS="${DOOR_BG_COOLDOWN_MS:-350}"
export DOOR_MAX_CLIP_FRAMES="${DOOR_MAX_CLIP_FRAMES:-4}"
export DOOR_COOLDOWN_MS="${DOOR_COOLDOWN_MS:-900}"
export DOOR_BG_AUDIO_SECONDS="${DOOR_BG_AUDIO_SECONDS:-2.2}"
export DOOR_CAMERA_INPUT_FPS="${DOOR_CAMERA_INPUT_FPS:-10}"
export DOOR_CAMERA_ONE_SHOT="${DOOR_CAMERA_ONE_SHOT:-1}"
export DOOR_CAMERA_OUTPUT_FPS="${DOOR_CAMERA_OUTPUT_FPS:-12}"
export DOOR_CAMERA_FRAME_TIMEOUT_MS="${DOOR_CAMERA_FRAME_TIMEOUT_MS:-1400}"
export DOOR_PREVIEW_INTERVAL_MS="${DOOR_PREVIEW_INTERVAL_MS:-120}"
export DOOR_PREVIEW_TTL_MS="${DOOR_PREVIEW_TTL_MS:-3000}"
export CAMERA_LOCK_DIR="${CAMERA_LOCK_DIR:-/workspace/identification/locks}"
export PIPELINE_AUTO_START="${PIPELINE_AUTO_START:-1}"
export VISION_CPU_LIMIT="${VISION_CPU_LIMIT:-$(count_cpuset_cpus "$VISION_CPUSET")}"
export AUDIO_CPU_LIMIT="${AUDIO_CPU_LIMIT:-$(count_cpuset_cpus "$AUDIO_CPUSET")}"
export GATEWAY_CPU_LIMIT="${GATEWAY_CPU_LIMIT:-$(count_cpuset_cpus "$GATEWAY_CPUSET")}"
export DOOR_AGENT_CPU_LIMIT="${DOOR_AGENT_CPU_LIMIT:-$(count_cpuset_cpus "$DOOR_AGENT_CPUSET")}"

echo "[INFO] CPU: logical=$CPU_THREADS, physical=$PHYSICAL_CORES"
echo "[INFO] Runtime threads: VISION_INTRA_THREADS=$VISION_INTRA_THREADS, AUDIO_INTRA_THREADS=$AUDIO_INTRA_THREADS"
echo "[INFO] CPU pinning: VISION_CPUSET=$VISION_CPUSET, AUDIO_CPUSET=$AUDIO_CPUSET, GATEWAY_CPUSET=$GATEWAY_CPUSET, DOOR_AGENT_CPUSET=$DOOR_AGENT_CPUSET"
echo "[INFO] CPU quotas: VISION_CPU_LIMIT=$VISION_CPU_LIMIT, AUDIO_CPU_LIMIT=$AUDIO_CPU_LIMIT, GATEWAY_CPU_LIMIT=$GATEWAY_CPU_LIMIT, DOOR_AGENT_CPU_LIMIT=$DOOR_AGENT_CPU_LIMIT"

GPU_ENABLED=false
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
  GPU_ENABLED=true
  echo "[INFO] NVIDIA GPU обнаружена и доступна в Docker."
else
  echo "[WARN] GPU не найдена/не проброшена. Переключаем worker'ы в CPU режим."
  export RUNTIME_MODE=cpu
  export VISION_FORCE_CPU=1
  export AUDIO_FORCE_CPU=1
  export AUDIO_USE_CUDA=0
fi

if [ "$GPU_ENABLED" = true ] && [ "$RUNTIME_MODE" != "cpu" ]; then
  export VISION_FORCE_CPU=0
  export AUDIO_FORCE_CPU=0
  export AUDIO_USE_CUDA=1
fi

persist_runtime_env() {
  TMP_RUNTIME_ENV="$(mktemp)"
  if [ -f "$RUNTIME_ENV_FILE" ]; then
    grep -Ev '^(# Saved by start_docker\.sh|RUNTIME_MODE|VISION_FORCE_CPU|AUDIO_FORCE_CPU|AUDIO_USE_CUDA|VISION_CUDA_MEM_LIMIT_MB|AUDIO_CUDA_MEM_LIMIT_MB|VISION_INTRA_THREADS|AUDIO_INTRA_THREADS|VISION_INTER_THREADS|AUDIO_INTER_THREADS|OPENBLAS_NUM_THREADS|VISION_CPUSET|AUDIO_CPUSET|GATEWAY_CPUSET|DOOR_AGENT_CPUSET|VISION_CPU_LIMIT|AUDIO_CPU_LIMIT|GATEWAY_CPU_LIMIT|DOOR_AGENT_CPU_LIMIT)=' "$RUNTIME_ENV_FILE" > "$TMP_RUNTIME_ENV" || true
  fi
  cat >> "$TMP_RUNTIME_ENV" <<EOF
RUNTIME_MODE=$RUNTIME_MODE
VISION_FORCE_CPU=${VISION_FORCE_CPU:-0}
AUDIO_FORCE_CPU=${AUDIO_FORCE_CPU:-0}
AUDIO_USE_CUDA=${AUDIO_USE_CUDA:-0}
VISION_CUDA_MEM_LIMIT_MB=${VISION_CUDA_MEM_LIMIT_MB:-1024}
AUDIO_CUDA_MEM_LIMIT_MB=${AUDIO_CUDA_MEM_LIMIT_MB:-256}
VISION_INTRA_THREADS=$VISION_INTRA_THREADS
AUDIO_INTRA_THREADS=$AUDIO_INTRA_THREADS
VISION_INTER_THREADS=$VISION_INTER_THREADS
AUDIO_INTER_THREADS=$AUDIO_INTER_THREADS
OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS
VISION_CPUSET=$VISION_CPUSET
AUDIO_CPUSET=$AUDIO_CPUSET
GATEWAY_CPUSET=$GATEWAY_CPUSET
DOOR_AGENT_CPUSET=$DOOR_AGENT_CPUSET
VISION_CPU_LIMIT=$VISION_CPU_LIMIT
AUDIO_CPU_LIMIT=$AUDIO_CPU_LIMIT
GATEWAY_CPU_LIMIT=$GATEWAY_CPU_LIMIT
DOOR_AGENT_CPU_LIMIT=$DOOR_AGENT_CPU_LIMIT
EOF
  mv "$TMP_RUNTIME_ENV" "$RUNTIME_ENV_FILE"
  echo "[INFO] Актуальная runtime-конфигурация сохранена в $RUNTIME_ENV_FILE"
}

echo "--- [2/4] Сборка базового образа ---"
if docker buildx version >/dev/null 2>&1; then
  docker buildx build --load -t "$BASE_IMAGE" -f Dockerfile .
else
  docker build -t "$BASE_IMAGE" -f Dockerfile .
fi

echo "--- [3/4] Предварительная сборка workspace ---"
PREBUILD_CONTAINER_NAME="${PREBUILD_CONTAINER_NAME:-identification-prebuild}"
cleanup_prebuild_container() {
  docker rm -f "$PREBUILD_CONTAINER_NAME" >/dev/null 2>&1 || true
}

cleanup_prebuild_container
trap cleanup_prebuild_container INT TERM EXIT
docker run --name "$PREBUILD_CONTAINER_NAME" --rm \
  -v "$(pwd)":/workspace/identification \
  -v cargo-registry:/usr/local/cargo/registry \
  -v cargo-git:/usr/local/cargo/git \
  -v ort-cache:/root/.cache \
  -e ORT_STRATEGY="$ORT_STRATEGY" \
  -w /workspace/identification \
  "$BASE_IMAGE" \
  bash -lc '
    set -e
    cargo build --release --workspace
    ort_lib_dir="$(find /root/.cache/ort.pyke.io -type f -name libonnxruntime_providers_shared.so -printf "%h\n" 2>/dev/null | head -n1 || true)"
    if [ -n "$ort_lib_dir" ]; then
      cp -aLf "$ort_lib_dir"/libonnxruntime*.so* /workspace/identification/target/release/ 2>/dev/null || true
      cp -aLf "$ort_lib_dir"/libonnxruntime*.so* /workspace/identification/target/release/deps/ 2>/dev/null || true
      echo "[INFO] ORT CUDA libs подготовлены из: $ort_lib_dir"
    else
      echo "[WARN] ORT CUDA libs не найдены в cache после сборки."
    fi
  '
trap - INT TERM EXIT

persist_runtime_env

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
