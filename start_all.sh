#!/bin/bash
set -euo pipefail

# --- Конфигурация путей ---
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# Цвета для вывода
G='\033[0;32m'
Y='\033[1;33m'
R='\033[0;31m'
NC='\033[0m'

echo -e "${G}=== Инициализация системы (Dynamic Load Mode) ===${NC}"

# --- 1. Проверка системных зависимостей ---
check_env() {
    echo -n "Проверка CUDA... "
    if [ -d "/opt/cuda" ]; then
        echo -e "${G}OK${NC}"
    else
        echo -e "${R}Ошибка: /opt/cuda не найден${NC}"; exit 1
    fi

    echo -n "Проверка ONNX Runtime... "
    if [ -f "/usr/lib/libonnxruntime.so" ]; then
        echo -e "${G}OK (/usr/lib)${NC}"
    else
        echo -e "${R}Ошибка: выполните 'sudo pacman -S onnxruntime-cuda'${NC}"; exit 1
    fi
}

# --- 2. Настройка окружения (Clean Environment) ---
# Указываем ORT грузить системную либу во время работы (runtime)
export ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# Пути к библиотекам CUDA и cuDNN для динамического линковщика
export CUDA_HOME="/opt/cuda"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/lib:$LD_LIBRARY_PATH"

# Настройки GPU и логирования
export CUDA_VISIBLE_DEVICES=0
export ONNX_RUNTIME_LOG_SEVERITY=0
unset ORT_STRATEGY # Снимаем, так как используем load-dynamic

# --- 3. Очистка старых процессов и "зомби" файлов ---
echo -e "${Y}Очистка окружения...${NC}"
pkill -f gateway-service || true
pkill -f vision-worker || true
pkill -f audio-worker || true
# Удаляем старые .so, которые могли быть скачаны cargo ранее
find "$ROOT_DIR/target" -name "libonnxruntime.so*" -delete 2>/dev/null || true
sleep 1

# --- 4. Запуск сервисов ---
check_env

# Vision Worker (Запуск из поддиректории для корректных путей к моделям)
echo -e "${Y}[1/3] Запуск Vision Worker...${NC}"
cd "$ROOT_DIR/vision-worker"
nohup cargo run > "$LOG_DIR/vision.log" 2>&1 &
echo -e "      Лог: logs/vision.log"

# Audio Worker
echo -e "${Y}[2/3] Запуск Audio Worker...${NC}"
cd "$ROOT_DIR/audio-worker"
nohup cargo run > "$LOG_DIR/audio.log" 2>&1 &
echo -e "      Лог: logs/audio.log"

# Gateway Service
echo -e "${Y}[3/3] Запуск Gateway Service...${NC}"
cd "$ROOT_DIR"
sleep 2 # Даем воркерам время на инициализацию GPU
nohup cargo run -p gateway-service > "$LOG_DIR/gateway.log" 2>&1 &
echo -e "      Лог: logs/gateway.log"

echo -e "\n${G}=== Система запущена успешно! ===${NC}"
echo "Для мониторинга GPU используй: watch -n 1 nvidia-smi"
echo "Для просмотра распознавания: tail -f logs/vision.log"

# Автоматический просмотр лога вижн-воркера
tail -f "$LOG_DIR/vision.log"