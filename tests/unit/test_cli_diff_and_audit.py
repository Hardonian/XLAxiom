"""Unit tests for expanded CLI diff, checkpoint diff, linting, and quantization audits."""

from __future__ import annotations

import argparse

import numpy as np

from xlaxiom.cli import command_diff, command_lint_checkpoint, command_quant_audit
from xlaxiom.safetensors_io import write_safetensors


def test_cli_diff_single_tensors(tmp_path) -> None:
    a_path = tmp_path / "actual.npy"
    d_path = tmp_path / "desired.npy"
    np.save(a_path, np.array([1.0, 2.0, 3.0]))
    np.save(d_path, np.array([1.0, 2.0, 3.0]))

    # Test positional paths
    args = argparse.Namespace(
        paths=[str(a_path), str(d_path)],
        actual=None,
        desired=None,
        rtol=1e-4,
        atol=1e-5,
        equal_nan=True,
        check_signed_zero=False,
        output=None,
        html=None,
        min_snr=None,
    )
    assert command_diff(args) == 0

    # Test divergence with html output
    html_path = tmp_path / "diff.html"
    np.save(d_path, np.array([1.0, 99.0, 3.0]))
    args_fail = argparse.Namespace(
        paths=[str(a_path), str(d_path)],
        actual=None,
        desired=None,
        rtol=1e-4,
        atol=1e-5,
        equal_nan=True,
        check_signed_zero=False,
        output=None,
        html=str(html_path),
        min_snr=None,
    )
    assert command_diff(args_fail) == 1
    assert html_path.exists()
    assert "XLAxiom Tensor Diff Visualizer" in html_path.read_text(encoding="utf-8")


def test_cli_diff_checkpoints(tmp_path) -> None:
    c1_path = tmp_path / "model1.safetensors"
    c2_path = tmp_path / "model2.safetensors"
    write_safetensors(c1_path, {"w1": np.ones((4, 4), dtype=np.float32)})
    write_safetensors(c2_path, {"w1": np.ones((4, 4), dtype=np.float32)})

    html_out = tmp_path / "ckpt_diff.html"
    args = argparse.Namespace(
        paths=[str(c1_path), str(c2_path)],
        actual=None,
        desired=None,
        rtol=1e-4,
        atol=1e-5,
        equal_nan=True,
        check_signed_zero=False,
        output=None,
        html=str(html_out),
        min_snr=None,
    )
    assert command_diff(args) == 0
    assert html_out.exists()
    assert "XLAxiom Checkpoint Parity Dashboard" in html_out.read_text(encoding="utf-8")


def test_cli_lint_checkpoint(tmp_path) -> None:
    clean_path = tmp_path / "clean.safetensors"
    corrupt_path = tmp_path / "corrupt.safetensors"

    write_safetensors(clean_path, {"w": np.ones((4, 4), dtype=np.float32)})
    write_safetensors(corrupt_path, {"w": np.array([1.0, float("nan")], dtype=np.float32)})

    args_clean = argparse.Namespace(
        checkpoint=str(clean_path),
        fail_on_warning=False,
        output=None,
    )
    assert command_lint_checkpoint(args_clean) == 0

    args_corrupt = argparse.Namespace(
        checkpoint=str(corrupt_path),
        fail_on_warning=False,
        output=None,
    )
    assert command_lint_checkpoint(args_corrupt) == 1


def test_cli_quant_audit_checkpoint(tmp_path) -> None:
    ckpt_path = tmp_path / "weights.safetensors"
    write_safetensors(ckpt_path, {"layer1": np.random.randn(8, 8).astype(np.float32)})

    args = argparse.Namespace(
        target=str(ckpt_path),
        tensor=None,
        dtype="fp8_e4m3",
        scale=None,
        percentile=99.9,
        output=None,
    )
    ret = command_quant_audit(args)
    assert ret in (0, 1)
