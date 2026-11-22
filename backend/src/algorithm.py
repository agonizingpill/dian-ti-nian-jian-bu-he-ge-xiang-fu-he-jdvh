"""Compatibility wrapper kept for legacy imports.

The versioned registry in :mod:`src.triage` is now the source of truth. This
module preserves the original v1 behaviour (seed 1066, depth 3) and tags every
result with its version code so old callers keep working unchanged.
"""
from .triage import BUILTIN_SPECS, FEATURE_NAMES, build_model


def classify(features: dict):
    model = build_model(BUILTIN_SPECS["tree-v1-1066"])
    result = model.classify(features)
    return result


def build_features(items):
    failed = [x for x in items if x["result"] == "fail"]
    scores = [float(x["score"]) for x in items if x["result"] != "na"]
    return {
        "critical_fail_count": sum(1 for x in failed if x["critical"]),
        "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "failed_item_count": len(failed),
        "documentation_gap": sum(1 for x in items if x["category"] == "documentation" and x["result"] == "fail"),
    }
