"""XLAxiom: evidence-first differential testing for numerical compiler workloads."""

from .execution import execute_array_api
from .model import SCHEMA_VERSION, Workload

__all__ = ["SCHEMA_VERSION", "Workload", "execute_array_api"]
__version__ = "0.6.0"
