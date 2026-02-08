CREATE TABLE IF NOT EXISTS access_logs (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id) ON DELETE SET NULL,
    device_id INT REFERENCES devices(id) ON DELETE SET NULL,
    granted BOOLEAN NOT NULL,
    details TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);