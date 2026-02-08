DROP TABLE IF EXISTS face_embeddings;
DROP TABLE IF EXISTS users CASCADE; -- Cascade to remove FKs if any
DROP TABLE IF EXISTS devices;
DROP TABLE IF EXISTS rooms;
DROP TABLE IF EXISTS zones CASCADE;
DROP TABLE IF EXISTS access_rules;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE face_embeddings (
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL
);

CREATE INDEX ON face_embeddings USING hnsw (embedding vector_cosine_ops);

CREATE TABLE zones (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE rooms (
    id SERIAL PRIMARY KEY,
    zone_id INT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    name TEXT NOT NULL
);

CREATE TABLE devices (
    id SERIAL PRIMARY KEY,
    room_id INT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    device_type TEXT NOT NULL DEFAULT 'camera',
    connection_string TEXT NOT NULL DEFAULT ''
);

CREATE TABLE access_rules (
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    zone_id INT NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, zone_id)
);