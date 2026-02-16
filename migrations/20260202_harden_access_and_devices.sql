-- Ensure canonical device types exist
INSERT INTO device_types (name)
VALUES ('camera'), ('lock'), ('microphone'), ('turnstile')
ON CONFLICT (name) DO NOTHING;

-- Data quality / uniqueness constraints
ALTER TABLE rooms
    ADD CONSTRAINT rooms_zone_id_name_key UNIQUE (zone_id, name);

ALTER TABLE devices
    ADD CONSTRAINT devices_room_id_name_key UNIQUE (room_id, name);

-- Helpful indexes for access checks
CREATE INDEX IF NOT EXISTS idx_access_rules_zones_user_id ON access_rules_zones(user_id);
CREATE INDEX IF NOT EXISTS idx_access_rules_zones_zone_id ON access_rules_zones(zone_id);
CREATE INDEX IF NOT EXISTS idx_access_rules_rooms_user_id ON access_rules_rooms(user_id);
CREATE INDEX IF NOT EXISTS idx_access_rules_rooms_room_id ON access_rules_rooms(room_id);
CREATE INDEX IF NOT EXISTS idx_rooms_zone_id ON rooms(zone_id);
CREATE INDEX IF NOT EXISTS idx_devices_room_id ON devices(room_id);
