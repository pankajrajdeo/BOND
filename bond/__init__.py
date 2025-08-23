from .runtime_env import configure_runtime
configure_runtime()  # ensure OpenMP env is set before any heavy imports

from .pipeline import BondMatcher

__all__ = ["BondMatcher"]
