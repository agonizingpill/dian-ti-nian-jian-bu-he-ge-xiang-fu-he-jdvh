"""Deterministic in-memory stubs for third-party modules used by the app.

The CI/sandbox image has no sklearn/pymysql/redis; these stubs implement just
enough of each API for the triage versioning guarantees to be exercised.
"""
from collections import Counter


class _Tree:
    def __init__(self, **params):
        self.max_depth = params.get("max_depth", 3)
        self.random_state = params.get("random_state")
        self._node = None
        self.classes_ = None

    def fit(self, X, y):
        rows = [list(v) + [lab] for v, lab in zip(X, y)]
        self.classes_ = sorted(set(y))
        self._node = _fit(rows, 0, self.max_depth, self.classes_)
        return self

    def predict(self, X):
        return [_predict(self._node, v) for v in X]

    def predict_proba(self, X):
        out = []
        for v in X:
            label = _predict(self._node, v)
            out.append([1.0 if c == label else 0.0 for c in self.classes_])
        return out

    def get_params(self):
        return {"max_depth": self.max_depth, "random_state": self.random_state}


def DecisionTreeClassifier(**params):
    return _Tree(**params)


def export_text(tree, feature_names):
    return _render(tree._node, feature_names, 0)


def _gini(rows):
    n = len(rows)
    counts = Counter(r[-1] for r in rows)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def _leaf(rows, classes):
    counts = Counter(r[-1] for r in rows)
    top = max(counts.values())
    return ("leaf", sorted(c for c, n in counts.items() if n == top)[0])


def _fit(rows, depth, max_depth, classes):
    if depth >= max_depth or len({r[-1] for r in rows}) == 1 or len(rows) == 1:
        return _leaf(rows, classes)
    parent = _gini(rows)
    best = None
    for f in range(len(rows[0]) - 1):
        vals = sorted({r[f] for r in rows})
        for t in ((a + b) / 2 for a, b in zip(vals, vals[1:])):
            left = [r for r in rows if r[f] <= t]
            right = [r for r in rows if r[f] > t]
            if not left or not right:
                continue
            gain = parent - (len(left) / len(rows)) * _gini(left) \
                - (len(right) / len(rows)) * _gini(right)
            if best is None or gain > best[0]:
                best = (gain, f, t, left, right)
    if best is None or best[0] <= 0:
        return _leaf(rows, classes)
    _, f, t, left, right = best
    return ("node", f, t,
            _fit(left, depth + 1, max_depth, classes),
            _fit(right, depth + 1, max_depth, classes))


def _predict(node, vec):
    while node[0] != "leaf":
        _, f, t, left, right = node
        node = left if vec[f] <= t else right
    return node[1]


def _render(node, names, depth):
    if node[0] == "leaf":
        return "|   " * depth + f"class: {node[1]}\n"
    _, f, t, left, right = node
    pad = "|   " * depth
    return (f"{pad}|--- {names[f]} <= {t:.2f}\n"
            + _render(left, names, depth + 1)
            + f"{pad}|--- {names[f]} >  {t:.2f}\n"
            + _render(right, names, depth + 1))
