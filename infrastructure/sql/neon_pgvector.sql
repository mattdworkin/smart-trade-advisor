CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_source
    ON knowledge_documents (source);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_created_at
    ON knowledge_documents (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_embedding
    ON knowledge_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    user_role TEXT NOT NULL,
    question TEXT NOT NULL,
    sources_used JSONB NOT NULL DEFAULT '[]'::jsonb,
    actions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
