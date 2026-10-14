"""XLAxiom: evidence-first differential testing for numerical compiler workloads."""

from ._version import __version__
from .execution import execute_array_api
from .model import SCHEMA_VERSION, Workload

__all__ = ["SCHEMA_VERSION", "Workload", "__version__", "execute_array_api"]
