"""Unit tests for universal testing interceptor and rich assertion enhancer."""

from __future__ import annotations

import numpy as np
import pytest

from xlaxiom.assertions import NumericalAssertionError
from xlaxiom.patch_testing import disable, enable, is_enabled, patch


def test_patch_numpy_assert_allclose(tmp_path) -> None:
    assert not is_enabled()

    with patch(save_html=True, output_dir=tmp_path):
        assert is_enabled()

        # Case 1: Tensors match
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_allclose(a, b)

        # Case 2: Tensors diverge
        c = np.array([1.0, 5.0, 3.0], dtype=np.float32)
        with pytest.raises(NumericalAssertionError) as exc_info:
            np.testing.assert_allclose(a, c, rtol=1e-5, atol=1e-5)

        err_msg = str(exc_info.value)
        assert "numpy.testing.assert_allclose (XLAxiom)" in err_msg
        assert "Signal-to-Noise Ratio (SNR)" in err_msg
        assert "Cosine similarity" in err_msg
        assert exc_info.value.diff is not None
        assert exc_info.value.diff.mismatched_elements == 1

        # Verify HTML report generated
        html_files = list(tmp_path.glob("numpy_assert_*.html"))
        assert len(html_files) >= 1
        assert "XLAxiom Assertion Interception" in html_files[0].read_text(encoding="utf-8")

    assert not is_enabled()


def test_patch_torch_assert_close_if_available(tmp_path) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not installed")

    with patch(save_html=False):
        t1 = torch.tensor([1.0, 2.0, 3.0])
        t2 = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(t1, t2)

        t3 = torch.tensor([1.0, 9.9, 3.0])
        with pytest.raises(NumericalAssertionError) as exc_info:
            torch.testing.assert_close(t1, t3)

        assert "torch.testing.assert_close (XLAxiom)" in str(exc_info.value)
        assert exc_info.value.diff is not None
        assert exc_info.value.diff.mismatched_elements == 1


def test_enable_and_disable_lifecycle() -> None:
    assert not is_enabled()
    enable()
    assert is_enabled()
    disable()
    assert not is_enabled()

