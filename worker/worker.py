import json
import os
import time
from datetime import datetime

import pymysql
import redis

from triage_engine import FALLBACK_VERSION, SpecLoadError, TriageEngine

DB = {
    "host": os.getenv("MYSQL_HOST", "db"), "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "lift_user"), "password": os.getenv("MYSQL_PASSWORD", "lift_pass"),
    "database": os.getenv("MYSQL_DATABASE", "lift_review"), "cursorclass": pymysql.cursors.DictCursor, "autocommit": True,
}
rdb = redis.from_url(os.getenv("REDIS_URL", "redis://redis:6379/0"), decode_responses=True)


def db_conn():
    return pymysql.connect(**DB)


engine = TriageEngine(db_conn)


def claim_job(job_id):
    """Compare-and-set claim: duplicate queue deliveries cannot double-run."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE analysis_jobs SET status='running' WHERE id=%s AND status='queued'",
                (job_id,),
            )
            if cur.rowcount != 1:
                return None
            cur.execute("SELECT * FROM analysis_jobs WHERE id=%s", (job_id,))
            return cur.fetchone()


def process(job_id):
    job = claim_job(job_id)
    if not job:
        # Already claimed/completed by another delivery: never produce a
        # second conclusion for the same review sheet.
        return
    # Version pinned by the backend at intake; retries reuse exactly this tree.
    version = job.get("model_version") or FALLBACK_VERSION
    try:
        features = job["features_json"] if isinstance(job["features_json"], dict) else json.loads(job["features_json"])
        result = engine.classify(version, features)
        conclusion = {
            "rules": result["path_rules"],
            "model": result["model"],
            "model_version": result["model_version"],
            "features": features,
        }
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE analysis_jobs
                       SET status='completed',prediction=%s,probability=%s,path_rules_json=%s,
                           model_version=%s,completed_at=%s
                       WHERE id=%s AND status='running'""",
                    (result["prediction"], result["probability"],
                     json.dumps(conclusion, ensure_ascii=False),
                     result["model_version"], datetime.utcnow(), job_id),
                )
                cur.execute(
                    "UPDATE lifts SET status=IF(status='under_review','rectification',status) WHERE id=%s",
                    (job["lift_id"],),
                )
    except SpecLoadError as exc:
        # Do not swap models: mark failed so a retry reruns the pinned version.
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE analysis_jobs SET status='failed',error=%s,completed_at=%s
                       WHERE id=%s AND status='running'""",
                    (f"分流树版本 {version} 加载失败，未切换模型：{exc}", datetime.utcnow(), job_id),
                )
    except Exception as exc:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE analysis_jobs SET status='failed',error=%s,completed_at=%s
                       WHERE id=%s AND status='running'""",
                    (str(exc), datetime.utcnow(), job_id),
                )


def recover_interrupted():
    """Re-queue and re-publish jobs left running by a previous worker process."""
    try:
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM analysis_jobs WHERE status='running' ORDER BY id"
                )
                ids = [row["id"] for row in cur.fetchall()]
                if ids:
                    placeholders = ",".join(["%s"] * len(ids))
                    cur.execute(
                        f"UPDATE analysis_jobs SET status='queued',error=NULL "
                        f"WHERE status='running' AND id IN ({placeholders})",
                        tuple(ids),
                    )
        if ids:
            for jid in ids:
                rdb.lpush("lift_analysis", str(jid))
            print(f"recovered {len(ids)} interrupted job(s): {', '.join(map(str, ids))}", flush=True)
    except Exception as exc:
        print(f"startup recovery skipped: {exc}", flush=True)


def main():
    recover_interrupted()
    while True:
        try:
            open("/shared/worker_healthy", "w").write(datetime.utcnow().isoformat())
            message = rdb.brpop("lift_analysis", timeout=5)
            if message:
                process(int(message[1]))
        except Exception as exc:
            print(f"worker loop error: {exc}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
