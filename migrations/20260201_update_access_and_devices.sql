-- Enforce single embedding per user
ALTER TABLE face_embeddings
    ADD PRIMARY KEY (user_id);

ALTER TABLE voice_embeddings
    ADD PRIMARY KEY (user_id);

-- Expand access rules to support zone-level and room-level access
CREATE TABLE IF NOT EXISTS access_rules_zones (
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    zone_id INT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, zone_id)
);

CREATE TABLE IF NOT EXISTS access_rules_rooms (
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    room_id INT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, room_id)
);

INSERT INTO access_rules_zones (user_id, zone_id, created_at)
SELECT user_id, zone_id, created_at
FROM access_rules
ON CONFLICT DO NOTHING;

DROP TABLE IF EXISTS access_rules;

-- Normalize device types
CREATE TABLE IF NOT EXISTS device_types (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

INSERT INTO device_types (name)
SELECT DISTINCT device_type
FROM devices
ON CONFLICT DO NOTHING;

ALTER TABLE devices
    ADD COLUMN device_type_id INT REFERENCES device_types(id);

UPDATE devices
SET device_type_id = device_types.id
FROM device_types
WHERE devices.device_type = device_types.name;

ALTER TABLE devices
    ALTER COLUMN connection_string SET NOT NULL,
    ALTER COLUMN device_type_id SET NOT NULL;

ALTER TABLE devices
    DROP COLUMN device_type;
