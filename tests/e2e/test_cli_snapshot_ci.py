"""End-to-end CLI tests for snapshot and enhanced multi-tensor ci-gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xlaxiom.cli import main


def test_cli_ci_gate_multi_tensor_and_junit_xml(tmp_path: Path, capsys) -> None:
    actual_path = tmp_path / "actual.npz"
    desired_path = tmp_path / "desired.npz"
    junit_path = tmp_path / "junit.xml"
    out_comment = tmp_path / "pr.md"

    t1 = np.ones((2, 2), dtype=np.float32)
    t2 = np.full((3,), 2.0, dtype=np.float32)

    np.savez(actual_path, layer1=t1, layer2=t2)
    np.savez(desired_path, layer1=t1, layer2=t2)

    ret = main(
        [
            "ci-gate",
            "--actual",
            str(actual_path),
            "--desired",
            str(desired_path),
            "--output",
            str(out_comment),
            "--junit-xml",
            str(junit_path),
        ]
    )
    assert ret == 0
    assert junit_path.is_file()
    assert out_comment.is_file()

    junit_text = junit_path.read_text(encoding="utf-8")
    assert '<testsuite name="xlaxiom-ci-gate" tests="2" failures="0"' in junit_text
    assert 'name="layer1"' in junit_text
    assert 'name="layer2"' in junit_text


def test_cli_snapshot_record_and_check(tmp_path: Path, capsys) -> None:
    golden_path = tmp_path / "golden.npz"
    tensor_file = tmp_path / "tensor.npy"

    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(tensor_file, arr)

    # Record
    ret_rec = main(["snapshot", "record", "--path", str(golden_path), "--input", str(tensor_file)])
    assert ret_rec == 0
    assert golden_path.is_file()

    # Check - pass
    ret_chk = main(["snapshot", "check", "--path", str(golden_path), "--input", str(tensor_file)])
    assert ret_chk == 0
    captured = capsys.readouterr()
    assert "PASSED" in captured.out

    # Check - fail with divergent array
    bad_file = tmp_path / "bad.npy"
    np.save(bad_file, np.array([1.0, 99.0, 3.0], dtype=np.float32))
    ret_chk_bad = main(["snapshot", "check", "--path", str(golden_path), "--input", str(bad_file)])
    assert ret_chk_bad == 1
    captured_bad = capsys.readouterr()
    assert "FAILED" in captured_bad.out
