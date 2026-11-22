"""Minimal pymysql stub: only the constants referenced by app code."""
from . import cursors  # noqa: F401


def connect(**kwargs):
    raise RuntimeError("pymysql stub: tests inject FakeDB via triage_store.connection")
