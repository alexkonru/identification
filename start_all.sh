
#!/bin/bash

# Kill existing processes
echo "Stopping existing services..."
pkill -f gateway-service
pkill -f vision-worker
pkill -f audio-worker
sleep 2

# Set Environment Variables for CUDA
# Found libcublasLt.so.12 in /usr/local/lib/ollama/cuda_v12/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/ollama/cuda_v12/
# NOTE: do not point ORT_DYLIB_PATH to CUDA toolkit dirs.
# ort expects path to onnxruntime shared libs, not CUDA runtime libraries.
unset ORT_DYLIB_PATH

# CUDA is enabled by default. Set VISION_FORCE_CPU=1 only for debugging fallback.
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
