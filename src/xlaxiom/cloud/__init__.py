"""Google Cloud Platform (GCP) and accelerator ecosystem integration for XLAxiom."""

from __future__ import annotations

from .storage import clear_mock_gcs, is_gcs_uri, parse_gcs_uri, read_gcs_text, write_gcs_text
from .tpu import (
    TPU_GENERATIONS,
    CloudTPUJobResult,
    CloudTPUManager,
    MultiHostTPUOrchestrator,
    TPUSpec,
)
from .vertex import VertexJobSpec, create_vertex_job_spec, generate_kfp_component_yaml
from .xpk import XPKWorkloadSpec, generate_xpk_workload_spec

__all__ = [
    "TPU_GENERATIONS",
    "CloudTPUJobResult",
    "CloudTPUManager",
    "MultiHostTPUOrchestrator",
    "TPUSpec",
    "VertexJobSpec",
    "XPKWorkloadSpec",
    "clear_mock_gcs",
    "create_vertex_job_spec",
    "generate_kfp_component_yaml",
    "generate_xpk_workload_spec",
    "is_gcs_uri",
    "parse_gcs_uri",
    "read_gcs_text",
    "write_gcs_text",
]

