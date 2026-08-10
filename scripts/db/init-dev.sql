-- Development database bootstrap.
--
-- Applied once, on first container start. Product schemas are created by
-- Alembic migrations, not here — this only installs extensions and the
-- per-product schemas that migrations expect to exist.

-- pgvector backs embedding storage for RAG retrieval (Phase 2 onward).
CREATE EXTENSION IF NOT EXISTS vector;
-- Trigram indexes for entity-name fuzzy matching during entity resolution.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- One schema per product keeps ownership explicit and makes it possible to
-- split a product onto its own database later without renaming tables.
CREATE SCHEMA IF NOT EXISTS platform;
CREATE SCHEMA IF NOT EXISTS marketmind;
CREATE SCHEMA IF NOT EXISTS atlas;
CREATE SCHEMA IF NOT EXISTS cfo_ai;
CREATE SCHEMA IF NOT EXISTS sentinel;
CREATE SCHEMA IF NOT EXISTS finops;
CREATE SCHEMA IF NOT EXISTS venture;

-- Separate database for the test suite so a test run cannot truncate dev data.
SELECT 'CREATE DATABASE fie_test OWNER fie'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'fie_test')\gexec
