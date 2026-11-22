"""Persistence for triage-tree versions and the active-version pointer.

The active pointer is a singleton row updated in one statement. Intake
resolves the version inside the same transaction/statement that inserts the
job, so concurrent order intake always pins a concrete, then-effective
version instead of reading a pointer that can change mid-request.
"""
from __future__ import annotations

import json
from datetime import datetime
from threading import Lock

from .db import connection
from .triage import BUILTIN_SPECS, FALLBACK_VERSION, TriageSpec, SpecLoadError, build_model

TRIAGE_TABLES = [
    """CREATE TABLE IF NOT EXISTS triage_versions (
        version VARCHAR(64) PRIMARY KEY,
        title VARCHAR(160) NOT NULL,
        description TEXT NOT NULL,
        spec_json JSON NOT NULL,
        built_in BOOLEAN NOT NULL DEFAULT FALSE,
        load_status ENUM('pending','ready','error') NOT NULL DEFAULT 'pending',
        load_error TEXT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME NULL
    )""",
    """CREATE TABLE IF NOT EXISTS triage_active_pointer (
        id INT PRIMARY KEY DEFAULT 1,
        active_version VARCHAR(64) NOT NULL,
        switched_by INT NULL,
        switched_at DATETIME NOT NULL,
        reason VARCHAR(500) NOT NULL,
        CONSTRAINT chk_triage_singleton CHECK (id = 1),
        FOREIGN KEY (active_version) REFERENCES triage_versions(version),
        FOREIGN KEY (switched_by) REFERENCES users(id) ON DELETE SET NULL
    )""",
]

_schema_lock = Lock()
_schema_ready = False


def ensure_schema():
    global _schema_ready
    with _schema_lock:
        if _schema_ready:
            return
        with connection() as conn:
            with conn.cursor() as cur:
                for statement in TRIAGE_TABLES:
                    cur.execute(statement)
                _ensure_model_version_column(cur)
        _schema_ready = True


def _ensure_model_version_column(cur):
    """Older databases predate per-job version pinning; migrate in place."""
    cur.execute("""
        SELECT COUNT(*) AS found FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'analysis_jobs'
          AND COLUMN_NAME = 'model_version'
    """)
    if cur.fetchone()["found"] == 0:
        cur.execute(
            "ALTER TABLE analysis_jobs ADD COLUMN model_version VARCHAR(64) NULL "
            "AFTER features_json"
        )


def seed_builtin_versions():
    """Register built-in versions and make sure a valid pointer exists."""
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            for spec in BUILTIN_SPECS.values():
                record = spec.to_record()
                cur.execute(
                    "SELECT version FROM triage_versions WHERE version=%s",
                    (spec.version,),
                )
                if cur.fetchone():
                    continue
                load_status, load_error = _probe(spec)
                cur.execute(
                    """INSERT INTO triage_versions
                       (version,title,description,spec_json,built_in,load_status,load_error,updated_at)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
                    (record["version"], record["title"], record["description"],
                     json.dumps(record["spec_json"], ensure_ascii=False),
                     record["built_in"], load_status, load_error, datetime.utcnow()),
                )
            cur.execute("SELECT COUNT(*) AS n FROM triage_active_pointer WHERE id=1")
            if cur.fetchone()["n"] == 0:
                cur.execute(
                    """INSERT INTO triage_active_pointer(id,active_version,switched_by,switched_at,reason)
                       VALUES (1,%s,NULL,%s,%s)""",
                    (FALLBACK_VERSION, datetime.utcnow(), "系统初始化：默认启用基线分流树"),
                )


def _probe(spec: TriageSpec):
    try:
        build_model(spec)
        return "ready", None
    except SpecLoadError as exc:
        return "error", str(exc)


def list_versions() -> list[dict]:
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT v.version,v.title,v.description,v.built_in,v.load_status,v.load_error,
                       v.created_at,v.updated_at,p.active_version IS NOT NULL AS is_active
                FROM triage_versions v
                LEFT JOIN triage_active_pointer p ON p.id=1 AND p.active_version=v.version
                ORDER BY v.created_at, v.version
            """)
            return list(cur.fetchall())


def get_version_row(version: str) -> dict | None:
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM triage_versions WHERE version=%s", (version,))
            return cur.fetchone()


def get_active_pointer() -> dict:
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT p.active_version,p.switched_by,p.switched_at,p.reason,u.name switched_by_name
                FROM triage_active_pointer p LEFT JOIN users u ON u.id=p.switched_by
                WHERE p.id=1
            """)
            return cur.fetchone()


def load_spec(version: str) -> TriageSpec:
    row = get_version_row(version)
    if not row:
        raise KeyError(version)
    payload = row["spec_json"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return TriageSpec(
        version=row["version"],
        title=row["title"],
        description=row["description"],
        feature_names=tuple(payload["feature_names"]),
        training_x=tuple(tuple(r) for r in payload["training_x"]),
        training_y=tuple(payload["training_y"]),
        params=dict(payload["params"]),
        built_in=bool(row["built_in"]),
    )


def register_version(spec: TriageSpec) -> dict:
    ensure_schema()
    load_status, load_error = _probe(spec)
    record = spec.to_record()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO triage_versions
                   (version,title,description,spec_json,built_in,load_status,load_error,updated_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE title=VALUES(title),description=VALUES(description),
                     spec_json=VALUES(spec_json),load_status=VALUES(load_status),
                     load_error=VALUES(load_error),updated_at=VALUES(updated_at)""",
                (record["version"], record["title"], record["description"],
                 json.dumps(record["spec_json"], ensure_ascii=False),
                 record["built_in"], load_status, load_error, datetime.utcnow()),
            )
    row = get_version_row(spec.version)
    row["load_error_message"] = load_error
    return row


def activate_version(version: str, actor_id: int, reason: str) -> dict:
    """Point the singleton pointer at ``version`` after validating it loads."""
    ensure_schema()
    row = get_version_row(version)
    if not row:
        raise KeyError(version)
    spec = load_spec(version)
    build_model(spec)  # raises SpecLoadError without touching the pointer
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE triage_versions SET load_status='ready',load_error=NULL,updated_at=%s WHERE version=%s",
                (datetime.utcnow(), version),
            )
            cur.execute(
                """INSERT INTO triage_active_pointer(id,active_version,switched_by,switched_at,reason)
                   VALUES (1,%s,%s,%s,%s)
                   ON DUPLICATE KEY UPDATE active_version=VALUES(active_version),
                     switched_by=VALUES(switched_by),switched_at=VALUES(switched_at),reason=VALUES(reason)""",
                (version, actor_id, datetime.utcnow(), reason),
            )
    return get_active_pointer()


def pin_job(lift_id: int, features: dict) -> int:
    """Insert a queued job with the version effective at this exact instant.

    The active pointer is read inside the INSERT statement so the version and
    the job row are chosen atomically; a switch can never land between the
    read and the write.
    """
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO analysis_jobs(lift_id,status,features_json,model_version)
                   SELECT %s,'queued',%s,COALESCE(
                     (SELECT active_version FROM triage_active_pointer WHERE id=1), %s)""",
                (lift_id, json.dumps(features, ensure_ascii=False), FALLBACK_VERSION),
            )
            return cur.lastrowid


def requeue_job(job_id: int) -> dict:
    """Re-enqueue an existing failed/interrupted job without changing version."""
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM analysis_jobs WHERE id=%s FOR UPDATE", (job_id,))
            job = cur.fetchone()
            if not job:
                raise KeyError(job_id)
            if job["status"] not in ("failed",):
                # Running rows are only requeued by startup recovery, not users.
                raise ValueError(job["status"])
            cur.execute(
                """UPDATE analysis_jobs SET status='queued',error=NULL,completed_at=NULL
                   WHERE id=%s AND status='failed'""",
                (job_id,),
            )
            if cur.rowcount != 1:
                raise ValueError(job["status"])
            return job


def list_jobs_with_versions(limit: int = 100) -> list[dict]:
    """Review sheets with their intake-pinned version, newest first."""
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT j.id,j.lift_id,j.status,j.prediction,j.model_version,j.created_at,j.completed_at,
                          l.asset_code
                   FROM analysis_jobs j JOIN lifts l ON l.id=j.lift_id
                   ORDER BY j.id DESC LIMIT %s""",
                (limit,),
            )
            rows = cur.fetchall()
    for row in rows:
        # Rows produced before versioning existed were computed by v1.
        row["model_version"] = row.get("model_version") or FALLBACK_VERSION
    return rows


def recent_feature_jobs(limit: int = 20) -> list[dict]:
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id,lift_id,features_json,model_version
                   FROM analysis_jobs ORDER BY id DESC LIMIT %s""",
                (limit,),
            )
            jobs = []
            for row in cur.fetchall():
                payload = row["features_json"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                jobs.append({"id": row["id"], "lift_id": row["lift_id"],
                             "features": payload, "model_version": row["model_version"]})
            return jobs


def reset_interrupted_jobs() -> list[int]:
    """Jobs left 'running' by a crashed worker go back to the queue.

    Their pinned model_version is untouched, so retries reuse the same tree.
    Returns the ids that were reset and need to be re-published to the queue.
    """
    ensure_schema()
    with connection() as conn:
        with conn.cursor() as cur:
            # Capture the interrupted rows first, then reset them, so the
            # returned set is exactly what this recovery touched.
            cur.execute("SELECT id FROM analysis_jobs WHERE status='running' ORDER BY id")
            ids = [r["id"] for r in cur.fetchall()]
            if not ids:
                return []
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"UPDATE analysis_jobs SET status='queued',error=NULL "
                f"WHERE status='running' AND id IN ({placeholders})",
                tuple(ids),
            )
            return ids
