"""Minimal redis stub supporting lazy client construction in app modules."""


class _Redis:
    def __init__(self, *args, **kwargs):
        self.queue = []

    def lpush(self, key, value):
        self.queue.append((key, value))

    def brpop(self, key, timeout=0):
        return None

    def ping(self):
        return True


def from_url(*args, **kwargs):
    return _Redis()
