#!/bin/bash
set -euo pipefail

# Запуск всей системы в Docker с поддержкой GPU.
# По умолчанию:
# - vision пытается работать на GPU,
# - audio работает на CPU (чтобы не отбирать VRAM у vision).

# Перед запуском проверяем, что Docker видит compose-плагин.
docker compose version >/dev/null

# Сборка образа и запуск сервисов в фоне.
docker compose up -d --build db vision-worker audio-worker gateway-service

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
