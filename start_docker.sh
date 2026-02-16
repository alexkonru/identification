#!/bin/bash
set -euo pipefail

# Запуск всей системы в Docker с поддержкой GPU.
# По умолчанию:
# - vision пытается работать на GPU,
# - audio работает на CPU (чтобы не отбирать VRAM у vision).

# Перед запуском проверяем, что Docker видит compose-плагин.
docker compose version >/dev/null

# Проверяем, что Docker Engine умеет выдавать GPU контейнерам.
# Без nvidia-container-toolkit будет ошибка вида:
# "could not select device driver \"\" with capabilities: [[gpu]]"
if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  cat <<'MSG'
[ERROR] Docker сейчас не может использовать GPU (NVIDIA runtime недоступен).

Что нужно сделать на хосте (CachyOS/Arch):
  1) Установить nvidia-container-toolkit.
  2) Настроить runtime для Docker и перезапустить docker.service.
  3) Проверить командой:
       docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

После этого снова запусти:
  ./start_docker.sh
MSG
  exit 1
fi

# Собираем базовый образ один раз (без повторной сборки для каждого сервиса).
docker build -t identification-rust-base:latest -f Dockerfile .

# Запуск сервисов в фоне (без дополнительной сборки через compose).
docker compose up -d --no-build db vision-worker audio-worker gateway-service

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
