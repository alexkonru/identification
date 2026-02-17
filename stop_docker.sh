#!/bin/bash
set -euo pipefail

# Остановка docker-compose окружения проекта.
# По умолчанию: останавливаем и удаляем контейнеры + сеть.
# С томами (ОПАСНО: удалит данные БД): ./stop_docker.sh --volumes

if [[ "${1:-}" == "--volumes" || "${1:-}" == "-v" ]]; then
  docker compose down -v --remove-orphans
else
  docker compose down --remove-orphans
fi

docker compose ps || true
