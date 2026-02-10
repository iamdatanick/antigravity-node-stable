
CREATE DATABASE IF NOT EXISTS antigravity;
USE antigravity;
CREATE TABLE IF NOT EXISTS memory_episodic (
    event_id BIGINT,
    tenant_id VARCHAR(64),
    timestamp DATETIME,
    session_id VARCHAR(64),
    actor VARCHAR(32),
    action_type VARCHAR(32),
    content TEXT,
    embedding ARRAY<FLOAT>
) ENGINE=OLAP
PRIMARY KEY (event_id, tenant_id)
DISTRIBUTED BY HASH(tenant_id)
PROPERTIES ("replication_num" = "1");

CREATE TABLE IF NOT EXISTS memory_semantic (
    doc_id VARCHAR(64),
    tenant_id VARCHAR(64),
    chunk_id INT,
    content TEXT,
    source_uri VARCHAR(256),
    embedding ARRAY<FLOAT>
) ENGINE=OLAP
PRIMARY KEY (doc_id, tenant_id, chunk_id)
DISTRIBUTED BY HASH(tenant_id)
PROPERTIES ("replication_num" = "1");

CREATE TABLE IF NOT EXISTS memory_procedural (
    skill_id VARCHAR(64),
    description TEXT,
    argo_template_yaml TEXT,
    success_rate FLOAT,
    embedding ARRAY<FLOAT>
) ENGINE=OLAP
PRIMARY KEY (skill_id)
DISTRIBUTED BY HASH(skill_id)
PROPERTIES ("replication_num" = "1");
