"""Stdlib-only tests for the triage versioning guarantees.

Run: python -m unittest discover -s tests -t .
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUBS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_stubs")
sys.path.insert(0, STUBS)
sys.path.insert(0, os.path.join(ROOT, "backend"))
sys.path.insert(0, os.path.join(ROOT, "worker"))

# Install third-party stubs before importing application modules.
import sklearn  # noqa: E402,F401
sys.modules.setdefault("sklearn", sklearn)

from src import triage, triage_store  # noqa: E402
from src.triage import (  # noqa: E402
    BUILTIN_SPECS, V1_VERSION, V2_VERSION, TriageSpec, build_model,
)
import triage_engine  # noqa: E402
import worker as worker_mod  # noqa: E402
from tests._stubs.fake_db import FakeDB  # noqa: E402


class TriageStoreTestBase(unittest.TestCase):
    def setUp(self):
        self.db = FakeDB()
        # Point the store's connection() context manager at the fake DB.
        triage_store.connection = lambda: self.db
        triage_store._schema_ready = False
        triage_store.seed_builtin_versions()

    def features(self, **overrides):
        base = {"critical_fail_count": 0, "average_score": 92.0,
                "failed_item_count": 0, "documentation_gap": 0}
        base.update(overrides)
        return base


class RegistryTests(TriageStoreTestBase):
    def test_builtin_versions_registered_and_v1_active_by_default(self):
        versions = {r["version"]: r for r in triage_store.list_versions()}
        self.assertEqual(set(versions), {V1_VERSION, V2_VERSION})
        self.assertTrue(all(r["load_status"] == "ready" for r in versions.values()))
        self.assertEqual(triage_store.get_active_pointer()["active_version"], V1_VERSION)

    def test_two_versions_diverge_on_realistic_intakes(self):
        v1 = build_model(BUILTIN_SPECS[V1_VERSION])
        v2 = build_model(BUILTIN_SPECS[V2_VERSION])
        vector = self.features()  # all-pass lift, e.g. XH-118 / LH-044
        self.assertEqual(v1.classify(vector)["prediction"], "ready")
        self.assertEqual(v2.classify(vector)["prediction"], "review")
        # Critical safety failures remain a hard hold in both trees.
        critical = self.features(critical_fail_count=2, failed_item_count=2,
                                 average_score=72.6)
        self.assertEqual(v1.classify(critical)["prediction"], "hold")
        self.assertEqual(v2.classify(critical)["prediction"], "hold")

    def test_broken_spec_cannot_be_activated_and_old_version_stays(self):
        broken = TriageSpec(
            version="tree-broken", title="坏树", description="",
            feature_names=tuple(triage.FEATURE_NAMES),
            training_x=((1, 2, 3, 4),),
            training_y=("ready", "hold"),  # row/label mismatch
            params={"max_depth": 3, "random_state": 1}, built_in=False,
        )
        row = triage_store.register_version(broken)
        self.assertEqual(row["load_status"], "error")
        with self.assertRaises(Exception):
            triage_store.activate_version("tree-broken", 7, "尝试启用坏树")
        self.assertEqual(triage_store.get_active_pointer()["active_version"], V1_VERSION)

    def test_switch_and_switch_back(self):
        triage_store.activate_version(V2_VERSION, 7, "主管试用新版")
        self.assertEqual(triage_store.get_active_pointer()["active_version"], V2_VERSION)
        triage_store.activate_version(V1_VERSION, 7, "新版不合适，切回基线")
        self.assertEqual(triage_store.get_active_pointer()["active_version"], V1_VERSION)
        reasons = [r["reason"] for r in self.db.tables["triage_active_pointer"]]
        self.assertEqual(len(reasons), 1)  # singleton pointer, history lands in audit_logs


class ConcurrencyTests(TriageStoreTestBase):
    def test_each_job_pins_version_effective_at_intake(self):
        # Two orders arrive while v1 is active.
        job_a = triage_store.pin_job(101, self.features())
        job_b = triage_store.pin_job(102, self.features())
        # Supervisor switches to v2; subsequent orders pin v2.
        triage_store.activate_version(V2_VERSION, 7, "试用新版")
        job_c = triage_store.pin_job(103, self.features())
        versions = {jid: self._job(jid)["model_version"]
                    for jid in (job_a, job_b, job_c)}
        self.assertEqual(versions[job_a], V1_VERSION)
        self.assertEqual(versions[job_b], V1_VERSION)
        self.assertEqual(versions[job_c], V2_VERSION)
        # Switching back never rewrites already-pinned sheets.
        triage_store.activate_version(V1_VERSION, 7, "切回")
        self.assertEqual(self._job(job_a)["model_version"], V1_VERSION)
        self.assertEqual(self._job(job_c)["model_version"], V2_VERSION)

    def test_concurrent_intake_under_switch_never_loses_version(self):
        # The atomic INSERT...SELECT means every queued row resolves to one
        # concrete pointer value; no row may ever carry a NULL version.
        ids = [triage_store.pin_job(lid, self.features()) for lid in range(10)]
        triage_store.activate_version(V2_VERSION, 7, "试用")
        ids += [triage_store.pin_job(lid, self.features()) for lid in range(10, 20)]
        pinned = [self._job(jid)["model_version"] for jid in ids]
        self.assertEqual(len(pinned), 20)
        self.assertTrue(all(v in (V1_VERSION, V2_VERSION) for v in pinned))
        self.assertIn(V1_VERSION, pinned)
        self.assertIn(V2_VERSION, pinned)

    def test_missing_pointer_falls_back_to_baseline(self):
        self.db.tables["triage_active_pointer"].clear()
        job_id = triage_store.pin_job(909, self.features())
        self.assertEqual(self._job(job_id)["model_version"], V1_VERSION)

    def _job(self, job_id):
        return next(r for r in self.db.tables["analysis_jobs"] if r["id"] == job_id)


class RetryAndRecoveryTests(TriageStoreTestBase):
    def test_failed_retry_keeps_pinned_version_and_no_new_row(self):
        triage_store.activate_version(V2_VERSION, 7, "试用")
        job_id = triage_store.pin_job(201, self.features())
        original = self._job(job_id)
        original.update(status="failed", error="boom")
        requeued = triage_store.requeue_job(job_id)
        self.assertEqual(requeued["model_version"], V2_VERSION)
        self.assertEqual(self._job(job_id)["status"], "queued")
        self.assertIsNone(self._job(job_id)["error"])
        self.assertEqual(len(self.db.tables["analysis_jobs"]), 1)  # no duplicate sheet

    def test_running_or_completed_job_cannot_be_requeued(self):
        job_id = triage_store.pin_job(202, self.features())
        with self.assertRaises(ValueError):
            triage_store.requeue_job(job_id)
        self._job(job_id)["status"] = "completed"
        with self.assertRaises(ValueError):
            triage_store.requeue_job(job_id)

    def test_interrupted_running_jobs_recover_with_same_version(self):
        triage_store.activate_version(V2_VERSION, 7, "试用")
        interrupted = triage_store.pin_job(301, self.features())
        fresh = triage_store.pin_job(302, self.features())
        self._job(interrupted)["status"] = "running"
        reset_ids = triage_store.reset_interrupted_jobs()
        self.assertEqual(reset_ids, [interrupted])
        self.assertEqual(self._job(interrupted)["status"], "queued")
        self.assertEqual(self._job(interrupted)["model_version"], V2_VERSION)
        self.assertEqual(self._job(fresh)["status"], "queued")

    def _job(self, job_id):
        return next(r for r in self.db.tables["analysis_jobs"] if r["id"] == job_id)


class WorkerSemanticsTests(TriageStoreTestBase):
    def _engine(self):
        import triage_engine
        return triage_engine.TriageEngine(lambda: self.db)

    def test_worker_uses_pinned_version_and_writes_versioned_conclusion(self):
        import triage_engine

        engine = self._engine()
        v1_result = engine.classify(V1_VERSION, self.features())
        v2_result = engine.classify(V2_VERSION, self.features())
        self.assertEqual(v1_result["prediction"], "ready")
        self.assertEqual(v2_result["prediction"], "review")
        self.assertEqual(v1_result["model_version"], V1_VERSION)
        self.assertIn(V1_VERSION, json.dumps(v1_result))

    def test_worker_falls_back_to_builtin_when_registry_unreachable(self):
        import triage_engine

        class Unreachable(FakeDB):
            def cursor(self):
                raise RuntimeError("db down")

        engine = triage_engine.TriageEngine(lambda: Unreachable())
        result = engine.classify(V1_VERSION, self.features())
        self.assertEqual(result["model_version"], V1_VERSION)
        self.assertEqual(result["prediction"], "ready")

    def test_unknown_version_fails_hard_instead_of_switching_model(self):
        engine = self._engine()
        with self.assertRaises(Exception):
            engine.classify("tree-ghost", self.features())

    def test_claim_guard_prevents_duplicate_conclusion(self):
        import worker as worker_mod

        worker_mod.db_conn = lambda: self.db
        job_id = triage_store.pin_job(401, self.features())
        first = worker_mod.claim_job(job_id)
        second = worker_mod.claim_job(job_id)  # redelivered queue message
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(self._job(job_id)["status"], "running")

    def test_worker_process_completes_with_pinned_version(self):
        import worker as worker_mod

        worker_mod.db_conn = lambda: self.db
        worker_mod.engine = self._engine()
        triage_store.activate_version(V2_VERSION, 7, "试用")
        job_id = triage_store.pin_job(501, self.features())
        worker_mod.process(job_id)
        job = self._job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["model_version"], V2_VERSION)
        self.assertEqual(job["prediction"], "review")
        stored = json.loads(job["path_rules_json"])
        self.assertEqual(stored["model_version"], V2_VERSION)
        # Re-delivering the same queue message changes nothing.
        worker_mod.process(job_id)
        self.assertEqual(len([j for j in self.db.tables["analysis_jobs"]]), 1)
        self.assertEqual(self._job(job_id)["status"], "completed")

    def test_worker_broken_pinned_spec_marks_failed_without_swap(self):
        import worker as worker_mod

        worker_mod.db_conn = lambda: self.db
        broken_engine = self._engine()
        worker_mod.engine = broken_engine
        job_id = triage_store.pin_job(601, self.features())
        self._job(job_id)["model_version"] = "tree-ghost"
        self._job(job_id)["status"] = "queued"
        worker_mod.process(job_id)
        job = self._job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["model_version"], "tree-ghost")
        self.assertIn("未切换模型", job["error"])

    def _job(self, job_id):
        return next(r for r in self.db.tables["analysis_jobs"] if r["id"] == job_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
