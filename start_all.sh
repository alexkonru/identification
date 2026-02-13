
#!/bin/bash

# Kill existing processes
echo "Stopping existing services..."
pkill -f gateway-service
pkill -f vision-worker
pkill -f audio-worker
sleep 2

# CUDA settings
# NOTE:
# - We do NOT override ORT_DYLIB_PATH here.
# - We do NOT force custom CUDA toolkit paths by default, because mismatched libs
#   can cause runtime errors like: cudaErrorSymbolNotFound.
unset ORT_DYLIB_PATH

# Optional: use CUDA libs from ollama bundle only when explicitly requested.
if [ "${USE_OLLAMA_CUDA_LIBS:-0}" = "1" ]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/ollama/cuda_v12/
fi

# CUDA is enabled by default. Set VISION_FORCE_CPU=1 only for fallback debugging.
export VISION_FORCE_CPU=${VISION_FORCE_CPU:-0}

echo "Starting Audio Worker..."
nohup cargo run -p audio-worker > audio.log 2>&1 &

echo "Starting Vision Worker (CUDA)..."
nohup cargo run -p vision-worker > vision.log 2>&1 &

echo "Starting Gateway Service..."
# Give workers a moment to bind ports
sleep 2
nohup cargo run -p gateway-service > gateway.log 2>&1 &

echo "Services started!"
echo "Check logs: tail -f *.log"
