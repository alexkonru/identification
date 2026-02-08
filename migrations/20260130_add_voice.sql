CREATE TABLE IF NOT EXISTS voice_embeddings (
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding vector(192) NOT NULL
);

CREATE INDEX ON voice_embeddings USING hnsw (embedding vector_cosine_ops);
