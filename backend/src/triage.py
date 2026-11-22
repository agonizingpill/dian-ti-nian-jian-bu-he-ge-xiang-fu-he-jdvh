"""Versioned triage-tree registry.

Each triage version stores a fully self-describing model spec (training set,
hyper-parameters, feature order). A job pins ``model_version`` at intake time,
so later activation switches never change the tree used by an already accepted
review job. See ``triage_store`` for persistence and ``triage_routes`` for the
supervisor-facing API.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from sklearn.tree import DecisionTreeClassifier, export_text

V1_VERSION = "tree-v1-1066"
V2_VERSION = "tree-v2-2026-trial"
FALLBACK_VERSION = V1_VERSION

FEATURE_NAMES = [
    "critical_fail_count",
    "average_score",
    "failed_item_count",
    "documentation_gap",
]

# v1: the regulator-mirrored training set shipped with the original workbench.
V1_TRAINING_X = [
    [0, 96, 0, 0], [0, 84, 1, 0], [0, 78, 2, 0],
    [0, 72, 2, 1], [1, 96, 1, 0], [1, 68, 3, 1],
    [0, 91, 0, 1], [0, 66, 3, 1], [2, 88, 2, 0],
]
V1_TRAINING_Y = ["ready", "ready", "review", "hold", "hold",
                 "hold", "review", "hold", "hold"]
V1_PARAMS = {"max_depth": 3, "random_state": 1066}

# v2: trial tree trained on an expanded, re-labelled set. Documentation-only
# gaps and low-score cases without critical failures move from hold to review,
# while genuine safety failures stay on hold. Verified to disagree with v1 on
# several realistic intakes (see scripts/verify_triage_versions.py).
V2_TRAINING_X = V1_TRAINING_X + [
    [0, 70, 2, 1], [0, 88, 1, 1], [1, 82, 2, 0],
    [1, 90, 1, 0], [0, 95, 0, 1], [0, 60, 4, 1],
]
V2_TRAINING_Y = V1_TRAINING_Y + [
    "review", "review", "review",
    "review", "review", "hold",
]
V2_PARAMS = {"max_depth": 3, "random_state": 2026}


@dataclass(frozen=True)
class TriageSpec:
    version: str
    title: str
    description: str
    feature_names: tuple
    training_x: tuple
    training_y: tuple
    params: dict
    built_in: bool

    def to_record(self):
        return {
            "version": self.version,
            "title": self.title,
            "description": self.description,
            "spec_json": {
                "feature_names": list(self.feature_names),
                "training_x": [list(row) for row in self.training_x],
                "training_y": list(self.training_y),
                "params": dict(self.params),
            },
            "built_in": self.built_in,
        }


BUILTIN_SPECS = {
    V1_VERSION: TriageSpec(
        version=V1_VERSION,
        title="分流树 v1（1066 基线）",
        description="原始基线模型，固定随机种子 1066、最大深度 3。",
        feature_names=tuple(FEATURE_NAMES),
        training_x=tuple(tuple(r) for r in V1_TRAINING_X),
        training_y=tuple(V1_TRAINING_Y),
        params=dict(V1_PARAMS),
        built_in=True,
    ),
    V2_VERSION: TriageSpec(
        version=V2_VERSION,
        title="分流树 v2（2026 试用版）",
        description="主管试用版：扩大训练集并重标，文档缺口与无关键项失败转人工复核，关键安全失败仍一票否决。",
        feature_names=tuple(FEATURE_NAMES),
        training_x=tuple(tuple(r) for r in V2_TRAINING_X),
        training_y=tuple(V2_TRAINING_Y),
        params=dict(V2_PARAMS),
        built_in=True,
    ),
}


class SpecLoadError(Exception):
    """Raised when a stored spec cannot build a usable classifier."""


class TriageModel:
    def __init__(self, spec: TriageSpec, classifier):
        self.spec = spec
        self._classifier = classifier

    def classify(self, features: dict) -> dict:
        vector = [[features[name] for name in self.spec.feature_names]]
        prediction = self._classifier.predict(vector)[0]
        probability = float(max(self._classifier.predict_proba(vector)[0]))
        rules = export_text(self._classifier,
                            feature_names=list(self.spec.feature_names))
        return {
            "prediction": prediction,
            "probability": probability,
            "path_rules": rules,
            "features": features,
            "model": (
                "DecisionTreeClassifier("
                f"random_state={self.spec.params.get('random_state')},"
                f"max_depth={self.spec.params.get('max_depth')})"
            ),
            "model_version": self.spec.version,
        }


def build_model(spec: TriageSpec) -> TriageModel:
    """Build a classifier from a spec; raises SpecLoadError on any problem."""
    try:
        if len(spec.training_x) != len(spec.training_y) or not spec.training_x:
            raise ValueError("training set is empty or row/label size mismatch")
        width = len(spec.feature_names)
        if any(len(row) != width for row in spec.training_x):
            raise ValueError("training row width does not match feature_names")
        classifier = DecisionTreeClassifier(**spec.params)
        classifier.fit([list(r) for r in spec.training_x],
                       list(spec.training_y))
        # Smoke test: every registered feature must be evaluable.
        probe = {name: 0 for name in spec.feature_names}
        model = TriageModel(spec, classifier)
        model.classify(probe)
        return model
    except SpecLoadError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise SpecLoadError(f"模型规格加载失败: {exc}") from exc


class TriageRegistry:
    """Process-local cache of built models, keyed by version code."""

    def __init__(self):
        self._models: dict[str, TriageModel] = {}
        self._lock = Lock()

    def get(self, spec: TriageSpec) -> TriageModel:
        with self._lock:
            model = self._models.get(spec.version)
            if model is None:
                model = build_model(spec)
                self._models[spec.version] = model
            return model

    def invalidate(self, version: str | None = None):
        with self._lock:
            if version is None:
                self._models.clear()
            else:
                self._models.pop(version, None)


registry = TriageRegistry()
