"""First-class pytest plugin for XLAxiom differential verification.

Exposes pytest markers, fixtures, and failure hooks that automatically generate
offline HTML reports and portable evidence packages on test regression.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from .generation import GenerationConfig, generate
from .model import Workload
from .reporting import write_html_report


def pytest_configure(config: pytest.Config) -> None:
    """Register custom XLAxiom markers in pytest."""
    config.addinivalue_line(
        "markers",
        "xlaxiom(rtol, atol, backends): mark a test as an XLAxiom numerical differential test",
    )
    config.addinivalue_line(
        "markers",
        "xlaxiom_metamorphic(properties): mark a test as verifying metamorphic invariants",
    )


@pytest.fixture
def xlaxiom_workload_factory() -> Any:
    """Fixture providing a deterministic Workload generator."""

    def _factory(
        seed: int = 42, dtype: str = "float32", preferred_op: str | None = None
    ) -> Workload:
        return generate(GenerationConfig(seed=seed, dtype=dtype, preferred_op=preferred_op))

    return _factory


@pytest.fixture
def xlaxiom_evidence_dir(tmp_path: Path) -> Path:
    """Fixture providing an isolated evidence directory."""
    evidence_dir = tmp_path / "xlaxiom_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    return evidence_dir


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    """Hook to capture test failures on xlaxiom-marked tests and emit HTML evidence."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        marker = item.get_closest_marker("xlaxiom")
        if marker:
            reports_dir = Path(".xlaxiom-reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = reports_dir / f"{item.name}-evidence.html"
            try:
                # If test has an attached evidence dictionary, write interactive HTML
                divergence = getattr(item, "_xlaxiom_divergence", None)
                if divergence and isinstance(divergence, dict):
                    write_html_report(report_file, divergence)
            except Exception:
                pass
