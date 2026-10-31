"""Unit tests for offline HTML dashboard and visualizer generator."""

from __future__ import annotations

import numpy as np

from xlaxiom.assertions import diff
from xlaxiom.ui import generate_visualizer_html


def test_generate_visualizer_html_passed() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    d = diff(arr, arr)
    html_doc = generate_visualizer_html(d, title="Custom Test Visualizer")

    assert "<!DOCTYPE html>" in html_doc
    assert "<title>Custom Test Visualizer</title>" in html_doc
    assert "PASSED" in html_doc
    assert "<svg" in html_doc
    assert "Max Absolute Error" in html_doc
    assert "Signal-to-Noise Ratio" in html_doc


def test_generate_visualizer_html_failed() -> None:
    actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    desired = np.array([1.0, 2.0, 10.0], dtype=np.float32)
    d = diff(actual, desired, rtol=1e-5, atol=1e-5)
    html_doc = generate_visualizer_html(d)

    assert "DISCREPANCY DETECTED" in html_doc
    assert "Standalone Minimal Reproduction Script" in html_doc
    assert "assert_close(actual, desired" in html_doc
    assert "7.000e+00" in html_doc or "7.0" in html_doc
