"""Version-aware triage engine for the worker.

The worker never decides which version a job uses — the backend pinned
``model_version`` on the job row at intake time. This module only resolves that
pinned version: first from the database registry, then from the built-in
fallback copy if the registry is unreachable. A broken pinned spec fails the
job instead of silently swapping trees, so retries keep the same model.
"""
from __future__ import annotations

import json
from threading import Lock

from sklearn.tree import DecisionTreeClassifier, export_text

FEATURE_NAMES = ["critical_fail_count", "average_score",
                 "failed_item_count", "documentation_gap"]

FALLBACK_VERSION = "tree-v1-1066"

BUILTIN = {
    "tree-v1-1066": {
        "feature_names": FEATURE_NAMES,
        "training_x": [
            [0, 96, 0, 0], [0, 84, 1, 0], [0, 78, 2, 0],
            [0, 72, 2, 1], [1, 96, 1, 0], [1, 68, 3, 1],
            [0, 91, 0, 1], [0, 66, 3, 1], [2, 88, 2, 0],
        ],
        "training_y": ["ready", "ready", "review", "hold", "hold",
                       "hold", "review", "hold", "hold"],
        "params": {"max_depth": 3, "random_state": 1066},
    },
    "tree-v2-2026-trial": {
        "feature_names": FEATURE_NAMES,
        "training_x": [
            [0, 96, 0, 0], [0, 84, 1, 0], [0, 78, 2, 0],
            [0, 72, 2, 1], [1, 96, 1, 0], [1, 68, 3, 1],
            [0, 91, 0, 1], [0, 66, 3, 1], [2, 88, 2, 0],
            [0, 70, 2, 1], [0, 88, 1, 1], [1, 82, 2, 0],
            [1, 90, 1, 0], [0, 95, 0, 1], [0, 60, 4, 1],
        ],
        "training_y": ["ready", "ready", "review", "hold", "hold",
                       "hold", "review", "hold", "hold",
                       "review", "review", "review",
                       "review", "review", "hold"],
        "params": {"max_depth": 3, "random_state": 2026},
    },
}


class SpecLoadError(Exception):
    pass


def build_model(version: str, spec: dict):
    if len(spec["training_x"]) != len(spec["training_y"]) or not spec["training_x"]:
        raise SpecLoadError(f"{version}: 训练集为空或行数与标签不匹配")
    width = len(spec["feature_names"])
    if any(len(row) != width for row in spec["training_x"]):
        raise SpecLoadError(f"{version}: 训练特征维度与 feature_names 不一致")
    classifier = DecisionTreeClassifier(**spec["params"])
    classifier.fit([list(r) for r in spec["training_x"]],
                   list(spec["training_y"]))
    return classifier


class TriageEngine:
    def __init__(self, db_conn_factory):
        self._db_conn = db_conn_factory
        self._cache: dict[str, object] = {}
        self._lock = Lock()

    def _load_db_spec(self, version: str):
        with self._db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT spec_json FROM triage_versions WHERE version=%s",
                            (version,))
                row = cur.fetchone()
        if not row:
            return None
        payload = row["spec_json"]
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload

    def _get_classifier(self, version: str):
        with self._lock:
            if version in self._cache:
                return self._cache[version]
            spec = None
            try:
                spec = self._load_db_spec(version)
            except Exception:
                # Registry unreachable: continue with the built-in copy when
                # we have one so order processing is not interrupted.
                spec = None
            if spec is None:
                if version not in BUILTIN:
                    raise SpecLoadError(f"{version}: 版本注册表中不存在且无内置副本")
                spec = BUILTIN[version]
            classifier = build_model(version, spec)
            self._cache[version] = classifier
            return classifier

    def classify(self, version: str, features: dict):
        classifier = self._get_classifier(version)
        names = FEATURE_NAMES
        vector = [[features[n] for n in names]]
        prediction = classifier.predict(vector)[0]
        probability = float(max(classifier.predict_proba(vector)[0]))
        rules = export_text(classifier, feature_names=names)
        params = classifier.get_params()
        return {
            "prediction": prediction,
            "probability": probability,
            "path_rules": rules,
            "model": (
                "DecisionTreeClassifier("
                f"random_state={params.get('random_state')},"
                f"max_depth={params.get('max_depth')})"
            ),
            "model_version": version,
        }
