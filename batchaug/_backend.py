"""Backend selection for batchaug: 'auto', 'pytorch', or 'triton'."""

_backend = "auto"


def set_backend(name: str) -> None:
    """Set the active backend."""
    global _backend
    if name not in ("auto", "pytorch", "triton"):
        raise ValueError(f"Unknown backend {name!r}, expected 'auto', 'pytorch', or 'triton'")
    _backend = name


def get_backend() -> str:
    """Return the current backend setting (may be 'auto')."""
    return _backend


def resolve_backend() -> str:
    """Return 'pytorch' or 'triton' (resolving 'auto')."""
    if _backend == "auto":
        try:
            import triton  # noqa: F401

            return "triton"
        except ImportError:
            return "pytorch"
    return _backend
