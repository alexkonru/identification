#!/bin/bash
set -euo pipefail

# --- Настройки ---
BASE_IMAGE="identification-rust-base:latest"
RUNTIME_ENV_FILE=".server_runtime.env"
export ORT_STRATEGY=download

echo "--- [1/4] Подготовка окружения ---"
if [ -f "$RUNTIME_ENV_FILE" ]; then
    set -a
    source "$RUNTIME_ENV_FILE"
    set +a
    echo "[INFO] Загружены настройки из $RUNTIME_ENV_FILE"
fi

# Проверка GPU
GPU_ENABLED=false
if docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi >/dev/null 2>&1; then
    GPU_ENABLED=true
    echo "[INFO] NVIDIA GPU обнаружена и доступна в Docker."
else
    echo "[WARN] GPU не найдена или драйверы не проброшены. Будет использован CPU."
fi

echo "--- [2/4] Сборка базового образа ---"
# Собираем образ, где есть Rust и зависимости
docker build -t "$BASE_IMAGE" -f Dockerfile .

echo "--- [3/4] Предварительная сборка проекта (Pre-build) ---"
# Это самый важный этап. Собираем проект один раз, чтобы избежать Lockup в Docker Compose.
# Мы используем временный контейнер, который скомпилирует всё в общую папку target.
docker run --rm \
    -v "$(pwd)":/workspace/identification \
    -v cargo-registry:/usr/local/cargo/registry \
    -e ORT_STRATEGY=download \
    -w /workspace/identification \
    "$BASE_IMAGE" \
    cargo build --release --workspace

echo "--- [4/4] Запуск сервисов через Docker Compose ---"
# Теперь запускаем всё. --no-build гарантирует, что мы используем результат шага выше.
# Мы убрали лимиты CPU в compose (ранее), чтобы всё прошло гладко.
docker compose up -d --no-build --force-recreate --remove-orphans

# Ожидание готовности Gateway
wait_gateway() {
    echo -n "[INFO] Ожидание запуска Gateway (127.0.0.1:50051)..."
    local retries=30
    while ! nc -z 127.0.0.1 50051 >/dev/null 2>&1; do
        sleep 2
        retries=$((retries - 1))
        echo -n "."
        if [ "$retries" -le 0 ]; then
            echo -e "\n[ERROR] Gateway не ответил за 60 секунд."
            return 1
        fi
    done
    echo -e " Готово!"
}

if wait_gateway; then
    echo "----------------------------------------------------"
    echo "Система успешно запущена!"
    echo "Gateway: http://127.0.0.1:50051"
    echo "Для мониторинга используй: docker compose logs -f"
    echo "----------------------------------------------------"
else
    echo "[CRITICAL] Ошибка при запуске. Проверь логи: docker compose logs gateway-service"
    exit 1
fi