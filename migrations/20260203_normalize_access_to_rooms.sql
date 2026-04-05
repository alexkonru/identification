-- Нормализация доступа: только комнаты, без таблицы зон
-- Устраняет циклическую зависимость и дублирование

-- Шаг 1: Переносим доступ к зонам в доступ к комнатам
INSERT INTO access_rules_rooms (user_id, room_id, created_at)
SELECT arz.user_id, r.id, arz.created_at
FROM access_rules_zones arz
JOIN rooms r ON r.zone_id = arz.zone_id
ON CONFLICT DO NOTHING;

-- Шаг 2: Удаляем избыточную таблицу
DROP TABLE IF EXISTS access_rules_zones;

-- Шаг 3: Удаляем неиспользуемые индексы
DROP INDEX IF EXISTS idx_access_rules_zones_user_id;
DROP INDEX IF EXISTS idx_access_rules_zones_zone_id;

-- Шаг 4: Добавляем вычисляемую зону в rooms для удобства запросов
-- Это денормализация ради производительности, зона вычисляется из room->zone
-- и не создаёт цикла, т.к. room_id - это первичный ключ rooms
