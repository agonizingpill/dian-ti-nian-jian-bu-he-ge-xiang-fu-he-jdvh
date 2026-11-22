"""In-memory MySQL stand-in supporting the SQL shapes used by triage_store."""
from __future__ import annotations

import json
from copy import deepcopy


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.rowcount = 0
        self.lastrowid = None
        self._rows = []
        self.description = None

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _table(name):
        return name.split()[0].lower().strip("`")

    def _find(self, table, key_field, key_value):
        for row in table:
            if row.get(key_field) == key_value:
                return row
        raise KeyError(key_value)

    # -- SQL subset --------------------------------------------------------
    def execute(self, sql, params=()):
        self.rowcount = 0
        self.lastrowid = None
        self._rows = []
        self.description = None
        text = " ".join(sql.split())
        p = list(params)

        if text.startswith("SELECT COUNT(*) AS found FROM information_schema.COLUMNS"):
            self._rows = [{"found": 1}]  # pretend migrations already applied
            return
        if text.startswith("ALTER TABLE"):
            self.rowcount = 0
            return
        if text.startswith("CREATE TABLE"):
            self.rowcount = 0
            return
        if text.startswith("SELECT COUNT(*) AS n FROM triage_active_pointer"):
            self._rows = [{"n": len(self.db.tables["triage_active_pointer"])}]
            return
        if text.startswith("SELECT version FROM triage_versions WHERE version=%s"):
            self._rows = [r for r in self.db.tables["triage_versions"]
                          if r["version"] == p[0]]
            return
        if text.startswith("SELECT * FROM triage_versions WHERE version=%s"):
            self._rows = [deepcopy(r) for r in self.db.tables["triage_versions"]
                          if r["version"] == p[0]]
            return
        if text.startswith("SELECT v.version"):
            active = self.db.active_version()
            self._rows = []
            for r in self.db.tables["triage_versions"]:
                row = deepcopy(r)
                row["is_active"] = r["version"] == active
                row["switched_by_name"] = None
                self._rows.append(row)
            return
        if text.startswith("SELECT p.active_version"):
            pointer = self.db.pointer()
            self._rows = [pointer] if pointer else []
            return
        if text.startswith("SELECT spec_json FROM triage_versions"):
            self._rows = [r for r in self.db.tables["triage_versions"]
                          if r["version"] == p[0]]
            return
        if text.startswith("SELECT id,lift_id,features_json,model_version FROM analysis_jobs"):
            self._rows = deepcopy(sorted(self.db.tables["analysis_jobs"],
                                         key=lambda r: -r["id"])[: p[0]])
            return
        if text.startswith("SELECT * FROM analysis_jobs WHERE id=%s FOR UPDATE"):
            self._rows = [deepcopy(r) for r in self.db.tables["analysis_jobs"]
                          if r["id"] == p[0]]
            return
        if text.startswith("SELECT * FROM analysis_jobs WHERE id=%s"):
            self._rows = [deepcopy(r) for r in self.db.tables["analysis_jobs"]
                          if r["id"] == p[0]]
            return
        if text.startswith("SELECT id FROM analysis_jobs WHERE status='running'"):
            self._rows = [{"id": r["id"]} for r in self.db.tables["analysis_jobs"]
                          if r["status"] == "running"]
            return
        if text.startswith("SELECT id FROM analysis_jobs WHERE status='queued'"):
            queued = [r for r in self.db.tables["analysis_jobs"] if r["status"] == "queued"]
            if "ORDER BY id DESC" in text:
                queued = sorted(queued, key=lambda r: -r["id"])
            self._rows = [{"id": r["id"]} for r in queued]
            return
        if text.startswith("SELECT model_version FROM analysis_jobs WHERE id=%s"):
            self._rows = [{"model_version": r["model_version"]}
                          for r in self.db.tables["analysis_jobs"] if r["id"] == p[0]]
            return

        if text.startswith("INSERT INTO triage_versions"):
            table = self.db.tables["triage_versions"]
            existing = next((r for r in table if r["version"] == p[0]), None)
            payload = json.loads(p[3])
            if existing and "ON DUPLICATE KEY UPDATE" in text:
                existing.update(title=p[1], description=p[2], spec_json=payload,
                                built_in=p[4], load_status=p[5], load_error=p[6],
                                updated_at=p[7])
                self.rowcount = 2
            elif existing:
                raise AssertionError("duplicate version")
            else:
                row = {"version": p[0], "title": p[1], "description": p[2],
                       "spec_json": payload, "built_in": p[4],
                       "load_status": p[5], "load_error": p[6],
                       "created_at": "ts", "updated_at": p[7]}
                table.append(row)
                self.rowcount = 1
            return
        if text.startswith("INSERT INTO triage_active_pointer"):
            table = self.db.tables["triage_active_pointer"]
            if table and "ON DUPLICATE KEY UPDATE" in text:
                table[0].update(active_version=p[0], switched_by=p[1],
                                switched_at=p[2], reason=p[3])
                self.rowcount = 2
            else:
                # Initial seed: (active_version, switched_at, reason), id literal.
                table.append({"id": 1, "active_version": p[0], "switched_by": None,
                              "switched_at": p[1], "reason": p[2]})
                self.rowcount = 1
            return
        if text.startswith("INSERT INTO analysis_jobs") and "SELECT" in text:
            # Atomic pin: version comes from the singleton pointer in one step,
            # falling back to the baseline when the pointer row is missing.
            version = self.db.active_version() or p[2]
            row = {"id": self.db.next_job_id(), "lift_id": p[0], "status": "queued",
                   "features_json": json.loads(p[1]), "model_version": version,
                   "prediction": None, "probability": None, "path_rules_json": None,
                   "error": None, "completed_at": None}
            self.db.tables["analysis_jobs"].append(row)
            self.lastrowid = row["id"]
            self.rowcount = 1
            return
        if text.startswith("UPDATE triage_versions SET load_status='error'"):
            row = next(r for r in self.db.tables["triage_versions"] if r["version"] == p[2])
            row.update(load_status="error", load_error=p[0], updated_at=p[1])
            self.rowcount = 1
            return
        if text.startswith("UPDATE triage_versions SET load_status='ready'"):
            row = next(r for r in self.db.tables["triage_versions"] if r["version"] == p[1])
            row.update(load_status="ready", load_error=None, updated_at=p[0])
            self.rowcount = 1
            return
        if text.startswith("UPDATE analysis_jobs SET status='queued'"):
            if "AND id IN" in text:
                id_list = [v for v in p]
                targets = [r for r in self.db.tables["analysis_jobs"]
                           if r["status"] == "running" and r["id"] in id_list]
            else:
                targets = [r for r in self.db.tables["analysis_jobs"]
                           if r["status"] in ("failed", "running")]
            for r in targets:
                r.update(status="queued", error=None, completed_at=None)
            self.rowcount = len(targets)
            return
        if text.startswith("UPDATE analysis_jobs SET status='running' WHERE id=%s AND status='queued'"):
            row = next((r for r in self.db.tables["analysis_jobs"]
                        if r["id"] == p[0] and r["status"] == "queued"), None)
            if row:
                row["status"] = "running"
                self.rowcount = 1
            return
        if text.startswith("UPDATE analysis_jobs SET status='completed'"):
            row = next(r for r in self.db.tables["analysis_jobs"] if r["id"] == p[5])
            if row["status"] != "running":
                return
            row.update(status="completed", prediction=p[0], probability=p[1],
                       path_rules_json=json.dumps(json.loads(p[2]), ensure_ascii=False),
                       model_version=p[3], completed_at=p[4])
            self.rowcount = 1
            return
        if text.startswith("UPDATE analysis_jobs SET status='failed'"):
            row = next(r for r in self.db.tables["analysis_jobs"] if r["id"] == p[2])
            if row["status"] != "running":
                return
            row.update(status="failed", error=p[0], completed_at=p[1])
            self.rowcount = 1
            return
        raise AssertionError(f"unhandled SQL: {text}")

    def fetchone(self):
        return deepcopy(self._rows.pop(0)) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return deepcopy(rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDB:
    def __init__(self):
        self.tables = {
            "triage_versions": [],
            "triage_active_pointer": [],
            "analysis_jobs": [],
            "users": [{"id": 7, "name": "主管周岚"}],
        }
        self._job_seq = 0

    def next_job_id(self):
        self._job_seq += 1
        return self._job_seq

    def active_version(self):
        pointer = self.pointer()
        return pointer["active_version"] if pointer else None

    def pointer(self):
        rows = self.tables["triage_active_pointer"]
        return deepcopy(rows[0]) if rows else None

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
