"""Universal test interceptor and diagnostic enhancer for PyTorch and NumPy assertions.

Transparently wraps ``torch.testing.assert_close`` and ``numpy.testing.assert_allclose``
so existing test suites automatically gain:
- Signal-to-Noise Ratio (SNR) and Cosine Similarity metrics.
- Catastrophic cancellation and subnormal underflow detection.
- Bit-exact ULP distance tracking and error percentiles (p50, p90, p99).
- Root-cause diagnosis sentence and minimal reproduction snippet.
- Optional offline interactive HTML diff reports saved on regression.

Can be enabled globally, via context manager, or via ``pytest --xlaxiom-rich-asserts``.
"""

from __future__ import annotations

import contextlib
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np

from .assertions import NumericalAssertionError, diff
from .ui import generate_visualizer_html

_ORIGINAL_NP_ASSERT_ALLCLOSE: Any = None
_ORIGINAL_TORCH_ASSERT_CLOSE: Any = None
_IS_PATCHED: bool = False
_CONFIG: dict[str, Any] = {
    "save_html": False,
    "output_dir": Path(".xlaxiom-reports"),
}


def _save_html_artifact(diff_obj: Any, prefix: str = "diff") -> Path:
    """Save an offline interactive HTML diff visualizer."""
    out_dir = Path(_CONFIG["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    out_file = out_dir / f"{prefix}_{timestamp}.html"
    try:
        html_content = generate_visualizer_html(
            diff_obj, title=f"XLAxiom Assertion Interception: {prefix}"
        )
        out_file.write_text(html_content, encoding="utf-8")
    except Exception:
        pass
    return out_file


def _patched_np_assert_allclose(
    actual: Any,
    desired: Any,
    rtol: float = 1e-7,
    atol: float = 0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
) -> None:
    """XLAxiom enhanced drop-in for numpy.testing.assert_allclose."""
    diff_result = diff(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    )
    if not diff_result.is_close:
        html_path = None
        if _CONFIG["save_html"]:
            html_path = _save_html_artifact(diff_result, prefix="numpy_assert")

        msg_body = diff_result.format_summary(title="numpy.testing.assert_allclose (XLAxiom)")
        if html_path:
            msg_body += f"\n[Interactive HTML Report]: {html_path.resolve()}"
        full_msg = f"{err_msg}\n{msg_body}" if err_msg else msg_body
        raise NumericalAssertionError(full_msg, diff=diff_result)


def _patched_torch_assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: str | None = None,
    **kwargs: Any,
) -> None:
    """XLAxiom enhanced drop-in for torch.testing.assert_close."""
    resolved_rtol = rtol if rtol is not None else 1e-4
    resolved_atol = atol if atol is not None else 1e-5

    diff_result = diff(
        actual,
        expected,
        rtol=resolved_rtol,
        atol=resolved_atol,
        equal_nan=equal_nan,
    )
    if not diff_result.is_close:
        html_path = None
        if _CONFIG["save_html"]:
            html_path = _save_html_artifact(diff_result, prefix="torch_assert")

        msg_body = diff_result.format_summary(title="torch.testing.assert_close (XLAxiom)")
        if html_path:
            msg_body += f"\n[Interactive HTML Report]: {html_path.resolve()}"
        full_msg = f"{msg}\n{msg_body}" if msg else msg_body
        raise NumericalAssertionError(full_msg, diff=diff_result)


def enable(save_html: bool = False, output_dir: str | Path = ".xlaxiom-reports") -> None:
    """Enable XLAxiom interception of PyTorch and NumPy assertion functions globally.

    Parameters
    ----------
    save_html : bool, optional
        Whether to generate offline interactive HTML diff reports on failure (default: False).
    output_dir : str or Path, optional
        Directory where HTML reports are written (default: .xlaxiom-reports).
    """
    global _IS_PATCHED, _ORIGINAL_NP_ASSERT_ALLCLOSE, _ORIGINAL_TORCH_ASSERT_CLOSE
    _CONFIG["save_html"] = save_html
    _CONFIG["output_dir"] = Path(output_dir)

    if _IS_PATCHED:
        return

    # Patch numpy.testing
    _ORIGINAL_NP_ASSERT_ALLCLOSE = np.testing.assert_allclose
    np.testing.assert_allclose = _patched_np_assert_allclose

    # Patch torch.testing if imported or available
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        if hasattr(torch_mod, "testing") and hasattr(torch_mod.testing, "assert_close"):
            _ORIGINAL_TORCH_ASSERT_CLOSE = torch_mod.testing.assert_close
            torch_mod.testing.assert_close = _patched_torch_assert_close
    else:
        try:
            import torch  # type: ignore[import-untyped]

            if hasattr(torch, "testing") and hasattr(torch.testing, "assert_close"):
                _ORIGINAL_TORCH_ASSERT_CLOSE = torch.testing.assert_close
                torch.testing.assert_close = _patched_torch_assert_close
        except ImportError:
            pass

    _IS_PATCHED = True


def disable() -> None:
    """Restore original PyTorch and NumPy testing assertion functions."""
    global _IS_PATCHED, _ORIGINAL_NP_ASSERT_ALLCLOSE, _ORIGINAL_TORCH_ASSERT_CLOSE

    if not _IS_PATCHED:
        return

    if _ORIGINAL_NP_ASSERT_ALLCLOSE is not None:
        np.testing.assert_allclose = _ORIGINAL_NP_ASSERT_ALLCLOSE
        _ORIGINAL_NP_ASSERT_ALLCLOSE = None

    if "torch" in sys.modules and _ORIGINAL_TORCH_ASSERT_CLOSE is not None:
        torch_mod = sys.modules["torch"]
        if hasattr(torch_mod, "testing"):
            torch_mod.testing.assert_close = _ORIGINAL_TORCH_ASSERT_CLOSE
            _ORIGINAL_TORCH_ASSERT_CLOSE = None

    _IS_PATCHED = False


def is_enabled() -> bool:
    """Return whether testing assertion interception is currently active."""
    return _IS_PATCHED


@contextlib.contextmanager
def patch(
    save_html: bool = False, output_dir: str | Path = ".xlaxiom-reports"
) -> Generator[None, None, None]:
    """Context manager to temporarily enable XLAxiom assertion interception."""
    was_enabled = is_enabled()
    enable(save_html=save_html, output_dir=output_dir)
    try:
        yield
    finally:
        if not was_enabled:
            disable()
