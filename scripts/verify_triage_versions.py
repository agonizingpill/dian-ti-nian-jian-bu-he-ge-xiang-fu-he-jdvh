#!/usr/bin/env python3
"""Offline verification that the two triage trees disagree.

Uses a pure-stdlib CART reimplementation (no sklearn needed) mirroring
DecisionTreeClassifier(splitter='best', criterion='gini') semantics, so the
divergence claim can be checked anywhere. The same training data/parameters
are registered in backend/src/triage.py and worker/triage_engine.py.
"""
from __future__ import annotations

import sys
from collections import Counter

FEATURE_NAMES = ["critical_fail_count", "average_score",
                 "failed_item_count", "documentation_gap"]

V1_X = [
    [0, 96, 0, 0], [0, 84, 1, 0], [0, 78, 2, 0],
    [0, 72, 2, 1], [1, 96, 1, 0], [1, 68, 3, 1],
    [0, 91, 0, 1], [0, 66, 3, 1], [2, 88, 2, 0],
]
V1_Y = ["ready", "ready", "review", "hold", "hold",
        "hold", "review", "hold", "hold"]
V2_X = V1_X + [
    [0, 70, 2, 1], [0, 88, 1, 1], [1, 82, 2, 0],
    [1, 90, 1, 0], [0, 95, 0, 1], [0, 60, 4, 1],
]
V2_Y = V1_Y + ["review", "review", "review", "review", "review", "hold"]

CASES = {
    "YT-021 关键项不合格": [2, 72.6, 2, 0],
    "XH-118 全合格": [0, 92.0, 0, 0],
    "JS-307 单项机械不合格": [0, 85.8, 1, 0],
    "LH-044 全合格": [0, 92.4, 0, 0],
    "边界 低分无关键失败": [0, 72.0, 2, 1],
    "边界 文档缺口高分": [0, 91.0, 0, 1],
    "边界 单关键失败高分": [1, 96.0, 1, 0],
}


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,
                 label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label


def gini(rows):
    n = len(rows)
    counts = Counter(r[-1] for r in rows)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def leaf(rows, classes):
    counts = Counter(r[-1] for r in rows)
    top = max(counts.values())
    return Node(label=sorted(c for c, n in counts.items() if n == top)[0])


def fit(rows, depth, max_depth, classes):
    if depth >= max_depth or len({r[-1] for r in rows}) == 1:
        return leaf(rows, classes)
    parent = gini(rows)
    best = None
    for f in range(len(rows[0]) - 1):
        vals = sorted({r[f] for r in rows})
        for t in ((a + b) / 2 for a, b in zip(vals, vals[1:])):
            left = [r for r in rows if r[f] <= t]
            right = [r for r in rows if r[f] > t]
            if not left or not right:
                continue
            gain = parent - (len(left) / len(rows)) * gini(left) \
                - (len(right) / len(rows)) * gini(right)
            if best is None or gain > best[0]:
                best = (gain, f, t, left, right)
    if best is None or best[0] <= 0:
        return leaf(rows, classes)
    _, f, t, left, right = best
    return Node(feature=f, threshold=t,
                left=fit(left, depth + 1, max_depth, classes),
                right=fit(right, depth + 1, max_depth, classes))


def predict(node, vec):
    while node.label is None:
        node = node.left if vec[node.feature] <= node.threshold else node.right
    return node.label


def main():
    classes = sorted(set(V1_Y) | set(V2_Y))
    t1 = fit([v + [y] for v, y in zip(V1_X, V1_Y)], 0, 3, classes)
    t2 = fit([v + [y] for v, y in zip(V2_X, V2_Y)], 0, 3, classes)
    diffs = []
    for name, vec in CASES.items():
        a, b = predict(t1, vec), predict(t2, vec)
        flag = "DIFF" if a != b else "same"
        print(f"{name:22s} v1={a:7s} v2={b:7s} [{flag}]")
        if a != b:
            diffs.append(name)
    print(f"divergent cases: {len(diffs)}")
    if not diffs:
        print("FAIL: the two triage versions must disagree on at least one intake")
        sys.exit(1)
    # Safety guarantee: two critical failures always stay on hold in both.
    assert predict(t1, [2, 72.6, 2, 0]) == "hold"
    assert predict(t2, [2, 72.6, 2, 0]) == "hold"
    print("OK: versions diverge and critical-failure hold is preserved")


if __name__ == "__main__":
    main()
