-- Initial database creation. Application tables are created by the backend
-- (src/db.py SCHEMA + triage_store.ensure_schema) so MySQL upgrades and the
-- per-job model_version column stay in one place.
CREATE DATABASE IF NOT EXISTS lift_review;
