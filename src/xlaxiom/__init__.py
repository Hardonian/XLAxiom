"""XLAxiom: evidence-first differential testing for numerical compiler workloads."""

from .execution import execute_array_api
from .model import SCHEMA_VERSION, Workload
from ._version import __version__

__all__ = ["SCHEMA_VERSION", "Workload", "execute_array_api"]
