CREATE DATABASE IF NOT EXISTS antigravity;
USE antigravity;
CREATE TABLE IF NOT EXISTS memory_episodic (
    doc_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    chunk_id INT NOT NULL,
    content TEXT,
    metadata JSON,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
) PRIMARY KEY (doc_id, tenant_id, chunk_id)
DISTRIBUTED BY HASH(doc_id) BUCKETS 1;

CREATE TABLE IF NOT EXISTS memory_semantic (
    doc_id VARCHAR(255) NOT NULL,
    chunk_id INT NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    vector ARRAY<FLOAT>,
    content TEXT,
    metadata JSON
) PRIMARY KEY (doc_id, chunk_id)
DISTRIBUTED BY HASH(doc_id) BUCKETS 1;

CREATE TABLE IF NOT EXISTS memory_procedural (
    step_id VARCHAR(255) PRIMARY KEY,
    workflow_id VARCHAR(255),
    action TEXT,
    result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
) DISTRIBUTED BY HASH(step_id) BUCKETS 1;
