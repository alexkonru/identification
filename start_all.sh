#!/bin/bash
set -euo pipefail

# Kill existing processes
"${PKILL_BIN:-pkill}" -f gateway-service >/dev/null 2>&1 || true
"${PKILL_BIN:-pkill}" -f vision-worker >/dev/null 2>&1 || true
"${PKILL_BIN:-pkill}" -f audio-worker >/dev/null 2>&1 || true
sleep 1

# Runtime toggles
export VISION_FORCE_CPU="${VISION_FORCE_CPU:-0}"
export VISION_CUDA_DEVICE_ID="${VISION_CUDA_DEVICE_ID:-0}"
export VISION_CUDA_MEM_LIMIT_MB="${VISION_CUDA_MEM_LIMIT_MB:-2048}"

# Avoid stale manually-pinned ORT library unless user explicitly wants it.
if [ "${USE_CUSTOM_ORT_DYLIB:-0}" != "1" ]; then
  unset ORT_DYLIB_PATH || true
fi

# Optional CUDA libs from Ollama bundle when requested.
if [ "${USE_OLLAMA_CUDA_LIBS:-0}" = "1" ] && [ -d "/usr/local/lib/ollama/cuda_v12" ]; then
  export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH:-}"
fi

# Make CUDA provider loading more predictable in mixed setups.
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

mkdir -p logs

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

start_service "audio" cargo run -p audio-worker
start_service "vision" cargo run -p vision-worker
sleep 2

if [ -f logs/vision.pid ]; then
  vision_pid="$(cat logs/vision.pid)"
  if ! is_pid_alive "$vision_pid"; then
    echo "Vision worker crashed during startup. See logs/vision.log"
    exit 1
  fi
fi

start_service "gateway" cargo run -p gateway-service
sleep 2

if [ -f logs/gateway.pid ]; then
  gateway_pid="$(cat logs/gateway.pid)"
  if ! is_pid_alive "$gateway_pid"; then
    echo "Gateway service crashed during startup. See logs/gateway.log"
    exit 1
  fi
fi

echo "Services started."
echo "  Audio log:   logs/audio.log"
echo "  Vision log:  logs/vision.log"
echo "  Gateway log: logs/gateway.log"
echo "Tip: tail -f logs/vision.log"
