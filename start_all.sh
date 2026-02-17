#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# Скрипт запуска всей системы (audio + vision + gateway)
# Цель:
#   1) максимально использовать GPU для vision,
#   2) не ронять систему при проблемах CUDA/ORT,
#   3) сохранить предсказуемый и диагностируемый запуск.
# ------------------------------------------------------------

# Загружаем сохранённые runtime-настройки (если есть),
# чтобы клиент мог централизованно менять режим запуска.
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-.server_runtime.env}"
if [ -f "$RUNTIME_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$RUNTIME_ENV_FILE"
  set +a
fi

# Останавливаем старые процессы, если они остались.
"${PKILL_BIN:-pkill}" -f gateway-service >/dev/null 2>&1 || true
"${PKILL_BIN:-pkill}" -f vision-worker >/dev/null 2>&1 || true
"${PKILL_BIN:-pkill}" -f audio-worker >/dev/null 2>&1 || true
sleep 1

# --------------------------
# Настройки Vision (GPU-first)
# --------------------------
# По умолчанию стартуем безопасно на CPU.
export VISION_FORCE_CPU="${VISION_FORCE_CPU:-1}"
# Идентификатор GPU.
# Если оставить пустым, vision-worker сам попробует несколько device_id (0..3)
# и выберет рабочий (полезно для гибридной графики).
export VISION_CUDA_DEVICE_ID="${VISION_CUDA_DEVICE_ID:-}"
# Лимит памяти CUDA EP для vision (в MB).
# Для MX230 (2GB) безопасно начинать с 1024..1536.
export VISION_CUDA_MEM_LIMIT_MB="${VISION_CUDA_MEM_LIMIT_MB:-1536}"

# --------------------------
# Настройки Audio
# --------------------------
# ВАЖНО: по умолчанию аудио оставляем на CPU, чтобы не "съедать" VRAM,
# которую лучше отдать vision-моделям.
# При желании можно включить GPU явно: AUDIO_USE_CUDA=1
export AUDIO_USE_CUDA="${AUDIO_USE_CUDA:-0}"
export AUDIO_FORCE_CPU="${AUDIO_FORCE_CPU:-1}"
# Лимит VRAM для audio CUDA EP (если AUDIO_USE_CUDA=1).
export AUDIO_CUDA_MEM_LIMIT_MB="${AUDIO_CUDA_MEM_LIMIT_MB:-256}"

# Если пользователь не попросил явно закрепить ORT_DYLIB_PATH,
# убираем его, чтобы не подхватить случайную несовместимую библиотеку.
if [ "${USE_CUSTOM_ORT_DYLIB:-0}" != "1" ]; then
  unset ORT_DYLIB_PATH || true
fi

# Опциональное подключение CUDA-библиотек из Ollama-bundle.
# Путь НЕ меняем, только используем при явном флаге.
if [ "${USE_OLLAMA_CUDA_LIBS:-0}" = "1" ] && [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
  export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

# LAZY обычно устойчивее на системах с несколькими наборами CUDA-библиотек.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"


# Делаем лог чище: оставляем полезные INFO по сервисам и WARN по onnxruntime.
# При необходимости можно переопределить вручную через RUST_LOG.
export RUST_LOG="${RUST_LOG:-vision_worker=info,audio_worker=info,gateway_service=info,ort::logging=warn}"

mkdir -p logs

# Небольшой preflight для диагностики: если nvidia-smi недоступен,
# просто пишем предупреждение и продолжаем (CPU fallback в коде сохранён).
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > logs/nvidia-smi.log 2>&1 || true
else
  echo "WARN: nvidia-smi not found" > logs/nvidia-smi.log
fi

start_service() {
  local name="$1"
  shift
  echo "Starting ${name}..."
  nohup "$@" > "logs/${name}.log" 2>&1 &
  echo "$!" > "logs/${name}.pid"
}

is_pid_alive() {
  local pid="$1"
  kill -0 "$pid" >/dev/null 2>&1
}

# Запускаем audio и vision раньше gateway,
# чтобы gateway подключался к уже живым backend-сервисам.
start_service "audio" cargo run -p audio-worker
start_service "vision" cargo run -p vision-worker
sleep 2

# Fail-fast: если vision умер на старте, сразу останавливаемся.
if [ -f logs/vision.pid ]; then
  vision_pid="$(cat logs/vision.pid)"
  if ! is_pid_alive "$vision_pid"; then
    echo "Vision worker crashed during startup. See logs/vision.log"
    exit 1
  fi
fi

start_service "gateway" cargo run -p gateway-service
sleep 2

# Fail-fast: если gateway умер, считаем запуск неуспешным.
if [ -f logs/gateway.pid ]; then
  gateway_pid="$(cat logs/gateway.pid)"
  if ! is_pid_alive "$gateway_pid"; then
    echo "Gateway service crashed during startup. See logs/gateway.log"
    exit 1
  fi
fi

echo "Services started."
echo "  Audio log:      logs/audio.log"
echo "  Vision log:     logs/vision.log"
echo "  Gateway log:    logs/gateway.log"
echo "  GPU snapshot:   logs/nvidia-smi.log"
echo "Tip: tail -f logs/vision.log"
