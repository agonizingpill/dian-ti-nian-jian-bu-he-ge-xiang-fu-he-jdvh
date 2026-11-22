import os
import sys

_STUB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _STUB_ROOT not in sys.path:
    sys.path.insert(0, _STUB_ROOT)

from sklearn_tree import DecisionTreeClassifier, export_text  # noqa: E402,F401
