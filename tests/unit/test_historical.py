"""Unit tests for xlaxiom.historical."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.historical import (
    HistoricalObservation,
    expected_hyperbolic_status,
    hyperbolic_gradient_study,
    main,
    replay_jax_40634,
)


class TestHistoricalObservation:
    def test_instantiation_and_fields(self) -> None:
        obs = HistoricalObservation(
            issue="https://github.com/jax-ml/jax/issues/40634",
            adapter="jax-40634-asinh-acosh-large-float32-v1",
            jax_version="0.4.30",
            jaxlib_version="0.4.30",
            platform="Linux-6.6.0-x86_64",
            input_float32=1e22,
            reference_gradient=1e-22,
            arcsinh_gradient=0.0,
            acosh_gradient=1e-22,
            classification="known_historical_regression_reproduced",
        )
        assert obs.issue == "https://github.com/jax-ml/jax/issues/40634"
        assert obs.adapter == "jax-40634-asinh-acosh-large-float32-v1"
        assert obs.input_float32 == 1e22
        assert obs.arcsinh_gradient == 0.0

    def test_immutability(self) -> None:
        obs = HistoricalObservation(
            issue="test_issue",
            adapter="test_adapter",
            jax_version="0.1.0",
            jaxlib_version="0.1.0",
            platform="Linux",
            input_float32=1.0,
            reference_gradient=1.0,
            arcsinh_gradient=1.0,
            acosh_gradient=1.0,
            classification="pass",
        )
        with pytest.raises((FrozenInstanceError, AttributeError)):
            obs.classification = "modified"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        obs = HistoricalObservation(
            issue="test_issue",
            adapter="test_adapter",
            jax_version="0.1.0",
            jaxlib_version="0.1.0",
            platform="Linux",
            input_float32=1.0,
            reference_gradient=1.0,
            arcsinh_gradient=1.0,
            acosh_gradient=1.0,
            classification="pass",
        )
        d = obs.to_dict()
        assert isinstance(d, dict)
        assert d["issue"] == "test_issue"
        assert d["adapter"] == "test_adapter"
        assert d["classification"] == "pass"


class TestExpectedHyperbolicStatus:
    def test_known_jax_versions(self) -> None:
        v0_10_2 = expected_hyperbolic_status("0.10.2")
        assert v0_10_2 is not None
        assert v0_10_2[("arcsinh", 1e22)] == "analytic_discrepancy"
        assert v0_10_2[("acosh", 1e22)] == "analytic_discrepancy"

        v0_11_1 = expected_hyperbolic_status("0.11.1")
        assert v0_11_1 is not None
        assert v0_11_1[("arcsinh", 1e22)] == "analytic_discrepancy"

        v0_11_2 = expected_hyperbolic_status("0.11.2")
        assert v0_11_2 is not None
        assert v0_11_2[("arcsinh", 1e22)] == "analytic_discrepancy"
        assert v0_11_2[("acosh", 1e22)] == "pass"

    def test_unknown_jax_version_returns_none(self) -> None:
        assert expected_hyperbolic_status("0.99.0") is None
        assert expected_hyperbolic_status("unknown_version") is None


class TestHyperbolicGradientStudy:
    def test_study_structure(self) -> None:
        study = hyperbolic_gradient_study()
        assert "study" in study
        assert "source" in study
        assert "jax" in study
        assert "jaxlib" in study
        assert "platform" in study
        assert "x64_enabled" in study
        assert "observations" in study

        obs_list = study["observations"]
        assert isinstance(obs_list, list)
        assert len(obs_list) == 3

        ops = [(item["operation"], item["input"]) for item in obs_list if isinstance(item, dict)]
        assert ops == [("arcsinh", 1e22), ("arcsinh", -1e22), ("acosh", 1e22)]

    def test_study_mocked_discrepancy(self) -> None:
        with patch("jax.grad") as mock_grad, patch("jax.jit") as mock_jit:
            mock_grad_fn = MagicMock()
            mock_grad_fn.return_value.block_until_ready.return_value = 0.0
            mock_grad.return_value = mock_grad_fn

            mock_jit_fn = MagicMock()
            mock_jit_fn.return_value.block_until_ready.return_value = 0.0
            mock_jit.return_value = mock_jit_fn

            study = hyperbolic_gradient_study()
            obs_list = study["observations"]
            assert isinstance(obs_list, list)
            for obs in obs_list:
                assert isinstance(obs, dict)
                assert obs["classification"] == "analytic_discrepancy"


class TestMainEntryPoint:
    def test_main_study_argv(self, capsys: pytest.CaptureFixture[str]) -> None:
        ret = main(["_study"])
        assert ret == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "study" in data
        assert "observations" in data

    def test_main_invalid_argv_raises(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["invalid_arg"])
        assert "historical module is invoked by xlaxiom historical" in str(exc_info.value)

    def test_main_none_argv_raises(self) -> None:
        with patch("sys.argv", ["historical"]):
            with pytest.raises(SystemExit) as exc_info:
                main(None)
            assert "historical module is invoked by xlaxiom historical" in str(exc_info.value)


class TestReplayJax40634:
    def test_replay_live_execution(self) -> None:
        obs = replay_jax_40634()
        assert isinstance(obs, HistoricalObservation)
        assert obs.issue == "https://github.com/jax-ml/jax/issues/40634"
        assert obs.adapter == "jax-40634-asinh-acosh-large-float32-v1"
        assert obs.classification in {
            "resolved_historical_regression",
            "known_historical_regression_reproduced",
        }

    def test_replay_mocked_resolved(self) -> None:
        import math

        reference = 1.0 / math.hypot(1.0, 1e22)
        with patch("jax.grad"), patch("jax.jit") as mock_jit:
            mock_compiled = MagicMock()
            mock_compiled.return_value.block_until_ready.return_value = reference
            mock_jit.return_value = mock_compiled

            obs = replay_jax_40634()
            assert obs.classification == "resolved_historical_regression"

    def test_replay_mocked_reproduced(self) -> None:
        with patch("jax.grad"), patch("jax.jit") as mock_jit:
            mock_compiled = MagicMock()
            mock_compiled.return_value.block_until_ready.return_value = 0.0
            mock_jit.return_value = mock_compiled

            obs = replay_jax_40634()
            assert obs.classification == "known_historical_regression_reproduced"
