"""Storage package — run history backends."""

from storage.run import Run
from storage.json_backend import JSONBackend

# Default backend used throughout the app
_backend: JSONBackend | None = None


def get_backend() -> JSONBackend:
    global _backend
    if _backend is None:
        _backend = JSONBackend()
    return _backend


def save_run(run: Run) -> None:
    try:
        get_backend().save(run)
    except Exception:
        pass  # Never crash the app over history persistence


def load_all_runs() -> list[Run]:
    try:
        return get_backend().load_all()
    except Exception:
        return []
