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
from .golden import UPDATE_ENV_VAR, check_golden
from .model import Workload
from .reporting import write_html_report


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the golden-file update and rich assertion switches."""
    parser.addoption(
        "--xlaxiom-update-golden",
        action="store_true",
        default=False,
        help=f"Rewrite XLAxiom golden files instead of comparing (or set {UPDATE_ENV_VAR}=1).",
    )
    parser.addoption(
        "--xlaxiom-rich-asserts",
        action="store_true",
        default=False,
        help=(
            "Supercharge torch.testing and numpy.testing assertions with rich XLAxiom diagnostics."
        ),
    )
    parser.addoption(
        "--xlaxiom-save-html",
        action="store_true",
        default=False,
        help="Save offline interactive HTML diff reports on assertion failure.",
    )
    parser.addini(
        "xlaxiom_rich_asserts",
        type="bool",
        default=False,
        help="Enable rich XLAxiom diagnostics for PyTorch and NumPy assertions.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom XLAxiom markers in pytest and configure rich assertions."""
    config.addinivalue_line(
        "markers",
        "xlaxiom(rtol, atol, backends): mark a test as an XLAxiom numerical differential test",
    )
    config.addinivalue_line(
        "markers",
        "xlaxiom_metamorphic(properties): mark a test as verifying metamorphic invariants",
    )
    rich_asserts = bool(config.getoption("--xlaxiom-rich-asserts")) or bool(
        config.getini("xlaxiom_rich_asserts")
    )
    if rich_asserts:
        save_html = bool(config.getoption("--xlaxiom-save-html"))
        from .patch_testing import enable as enable_rich_asserts

        enable_rich_asserts(save_html=save_html)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Tear down global hooks and restore testing assert functions."""
    from .patch_testing import disable as disable_rich_asserts

    disable_rich_asserts()


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


@pytest.fixture
def xlaxiom_golden(request: pytest.FixtureRequest) -> Any:
    """Golden-file checker: ``xlaxiom_golden("name", actual, rtol=..., atol=..., max_ulp=...)``.

    Files live in ``__xlaxiom_golden__/<test module>/<name>.npz`` beside the test file. A missing
    file fails the test; record it with ``pytest --xlaxiom-update-golden``.
    """
    root = Path(str(request.path)).parent / "__xlaxiom_golden__" / Path(str(request.path)).stem
    update = bool(request.config.getoption("--xlaxiom-update-golden")) or None

    def _check(name: str, actual: Any, **tolerances: Any) -> Any:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        return check_golden(root / f"{safe}.npz", actual, update=update, **tolerances)

    return _check


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[Any]
) -> Generator[None, Any, None]:
    """Hook to capture test failures on xlaxiom tests and emit HTML evidence."""
    outcome = yield
    report = outcome.get_result()

    if report.when == "call" and report.failed:
        reports_dir = Path(".xlaxiom-reports")
        # Check if exception is NumericalAssertionError
        if call.excinfo and hasattr(call.excinfo.value, "diff"):
            diff_obj = getattr(call.excinfo.value, "diff", None)
            if diff_obj and hasattr(diff_obj, "_repr_html_"):
                reports_dir.mkdir(parents=True, exist_ok=True)
                report_file = reports_dir / f"{item.name}-diff.html"
                try:
                    report_file.write_text(diff_obj._repr_html_(), encoding="utf-8")
                except Exception:
                    pass

        marker = item.get_closest_marker("xlaxiom")
        if marker:
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = reports_dir / f"{item.name}-evidence.html"
            try:
                # If test has an attached evidence dictionary, write interactive HTML
                divergence = getattr(item, "_xlaxiom_divergence", None)
                if divergence and isinstance(divergence, dict):
                    write_html_report(report_file, divergence)
            except Exception:
                pass
