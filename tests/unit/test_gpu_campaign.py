"""Unit tests for src/xlaxiom/gpu_campaign.py."""

from __future__ import annotations

import math

import pytest

from xlaxiom.comparison import Classification, ComparisonResult
from xlaxiom.execution import ExecutionResult, ExecutionStatus
from xlaxiom.gpu_campaign import (
    CampaignBudget,
    campaign_cases,
    cpu_vs_gpu,
    gpu_eager_vs_jit,
    replay_jax_40634_on_gpu,
    run_gpu_campaign,
    workload_families,
)
from xlaxiom.hardware import PhysicalGpu, redact_uuid
from xlaxiom.model import ComparisonPolicy


def make_device(
    name: str = "NVIDIA GeForce RTX 3060",
    sm: str = "8.6",
    driver: str = "580.178.04",
    used: int = 10,
    utilization: float = 0.0,
    ordinal: int = 0,
    smi_index: int = 0,
) -> PhysicalGpu:
    major, minor = (int(part) for part in sm.split("."))
    return PhysicalGpu(
        ordinal=ordinal,
        name=name,
        uuid_redacted=redact_uuid(f"GPU-{name}"),
        compute_capability=sm,
        sm=(major, minor),
        driver_version=driver,
        cuda_version="13.0",
        total_vram_mib=16384,
        used_vram_mib=used,
        free_vram_mib=16384 - used,
        utilization_percent=utilization,
        active_processes=(),
        smi_index=smi_index,
    )


class TestCampaignBudget:
    """Tests for CampaignBudget data class and validation."""

    def test_default_budget_is_valid(self) -> None:
        budget = CampaignBudget()
        assert budget.cases == 8
        assert budget.seed == 0
        assert budget.measurements == 5
        budget.validate()

    def test_custom_valid_budget(self) -> None:
        budget = CampaignBudget(cases=1, measurements=20)
        budget.validate()

        budget_max = CampaignBudget(cases=24, measurements=1)
        budget_max.validate()

    def test_invalid_cases_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="cases must be between 1 and 24"):
            CampaignBudget(cases=0).validate()

        with pytest.raises(ValueError, match="cases must be between 1 and 24"):
            CampaignBudget(cases=25).validate()

    def test_invalid_measurements_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="measurements must be between 1 and 20"):
            CampaignBudget(measurements=0).validate()

        with pytest.raises(ValueError, match="measurements must be between 1 and 20"):
            CampaignBudget(measurements=21).validate()


class TestCampaignCases:
    """Tests for workload families expansion and case generation."""

    def test_workload_families_returns_expected_motifs(self) -> None:
        families = workload_families()
        assert len(families) == 4
        names = [f["family"] for f in families]
        assert set(names) == {"matmul", "broadcasting", "reduction", "precision_sensitive"}

    def test_campaign_cases_small_budget(self) -> None:
        budget = CampaignBudget(cases=2)
        cases = campaign_cases(budget)
        assert len(cases) == 2
        assert cases[0]["family"] == "matmul"
        assert cases[0]["seed"] == 2
        assert cases[1]["family"] == "broadcasting"
        assert cases[1]["seed"] == 1

    def test_campaign_cases_wraps_and_derives_distinct_seeds(self) -> None:
        budget = CampaignBudget(cases=6)
        cases = campaign_cases(budget)
        assert len(cases) == 6
        # First 4 maintain base seeds
        assert cases[0]["seed"] == 2
        assert cases[1]["seed"] == 1
        assert cases[2]["seed"] == 3
        assert cases[3]["seed"] == 0
        # 5th case (index 4) wraps to family 0 ('matmul', base seed 2) -> seed = 2 + 4 = 6
        assert cases[4]["family"] == "matmul"
        assert cases[4]["seed"] == 6
        # 6th case (index 5) wraps to family 1 ('broadcasting', base seed 1) -> seed = 1 + 5 = 6
        assert cases[5]["family"] == "broadcasting"
        assert cases[5]["seed"] == 6

    def test_campaign_cases_invalid_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="cases must be between 1 and 24"):
            campaign_cases(CampaignBudget(cases=0))


class TestCpuVsGpu:
    """Tests for individual cpu_vs_gpu comparison records."""

    def test_cpu_vs_gpu_schema_and_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device()

        mock_execution = ExecutionResult(
            backend="numpy",
            mode="eager",
            status=ExecutionStatus.OK,
            outputs=(),
            device="cpu:0",
            environment={
                "requested_platform": "gpu",
                "actual_devices": "cpu:0",
                "device_index": "0",
            },
        )
        mock_comp = ComparisonResult(
            classification=Classification.PASS,
            explanation="pass",
            policy=ComparisonPolicy(),
            left=mock_execution.summary(),
            right=mock_execution.summary(),
        )

        monkeypatch.setattr("xlaxiom.execution.execute_numpy", lambda w: mock_execution)
        monkeypatch.setattr(
            "xlaxiom.execution.execute_jax",
            lambda w, jit, requested_platform, device_index: mock_execution,
        )
        monkeypatch.setattr(
            "xlaxiom.gpu_campaign.run_comparison",
            lambda w, mode, target: mock_comp,
        )
        monkeypatch.setattr(
            "xlaxiom.gpu_campaign._safe_benchmark",
            lambda w, m, p: {"median_ms": 1.5, "requested_platform": p},
        )

        result = cpu_vs_gpu(workload_seed=123, device=device, variant="cuda13", measurements=3)

        assert result["seed"] == 123
        assert result["variant"] == "cuda13"
        assert result["device"]["name"] == device.name
        assert result["device"]["ordinal"] == device.ordinal
        assert "numpy_vs_gpu_jit" in result
        assert "gpu_eager_vs_gpu_jit" in result
        assert result["statuses"]["numpy"] == "ok"
        assert result["timings"]["gpu_jit"]["median_ms"] == 1.5
        assert result["timings"]["cpu_jit"]["median_ms"] == 1.5

    def test_cpu_vs_gpu_handles_missing_cpu_baseline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device()

        mock_execution = ExecutionResult(
            backend="numpy",
            mode="eager",
            status=ExecutionStatus.OK,
            outputs=(),
            device="cpu:0",
        )
        mock_comp = ComparisonResult(
            classification=Classification.PASS,
            explanation="pass",
            policy=ComparisonPolicy(),
            left=mock_execution.summary(),
            right=mock_execution.summary(),
        )

        monkeypatch.setattr("xlaxiom.execution.execute_numpy", lambda w: mock_execution)
        monkeypatch.setattr(
            "xlaxiom.execution.execute_jax",
            lambda w, jit, requested_platform, device_index: mock_execution,
        )
        monkeypatch.setattr(
            "xlaxiom.gpu_campaign.run_comparison",
            lambda w, mode, target: mock_comp,
        )

        def fake_benchmark(w, m, platform):
            if platform == "cpu":
                return None
            return {"median_ms": 1.0}

        monkeypatch.setattr("xlaxiom.gpu_campaign._safe_benchmark", fake_benchmark)
        monkeypatch.setattr(
            "xlaxiom.gpu_campaign._cpu_baseline_subprocess", lambda m, payload: None
        )

        result = cpu_vs_gpu(workload_seed=1, device=device, variant="cuda12", measurements=2)

        assert result["timings"]["cpu_jit"] is None
        assert "cpu_note" in result["timings"]
        assert "isolated CUDA runtime exposes no CPU backend" in result["timings"]["cpu_note"]


class TestGpuEagerVsJit:
    """Tests for gpu_eager_vs_jit observation record."""

    def test_gpu_eager_vs_jit_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device(name="Tesla V100", ordinal=1)

        mock_comp = ComparisonResult(
            classification=Classification.PASS,
            explanation="pass",
            policy=ComparisonPolicy(),
            left={"device": "gpu:0", "environment": {"actual_devices": "gpu:0"}},
            right={"device": "gpu:0", "environment": {"actual_devices": "gpu:0"}},
        )

        monkeypatch.setattr(
            "xlaxiom.gpu_campaign.run_comparison",
            lambda w, mode, target: mock_comp,
        )

        result = gpu_eager_vs_jit(workload_seed=42, device=device, variant="cuda12")

        assert result["seed"] == 42
        assert result["device"] == "Tesla V100"
        assert result["ordinal"] == 1
        assert result["variant"] == "cuda12"
        assert result["classification"] == "pass"
        assert result["left_device"] == "gpu:0"
        assert result["actual_devices"]["left"] == "gpu:0"


class TestRunGpuCampaign:
    """Tests for the main run_gpu_campaign entrypoint."""

    def test_run_gpu_campaign_end_to_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device()

        def mock_cpu_vs_gpu(seed, dev, var, measurements):
            return {
                "seed": seed,
                "variant": var,
                "numpy_vs_gpu_jit": {"classification": "pass"},
            }

        def mock_gpu_eager_vs_jit(seed, dev, var):
            return {
                "seed": seed,
                "classification": "pass",
            }

        monkeypatch.setattr("xlaxiom.gpu_campaign.cpu_vs_gpu", mock_cpu_vs_gpu)
        monkeypatch.setattr("xlaxiom.gpu_campaign.gpu_eager_vs_jit", mock_gpu_eager_vs_jit)

        budget = CampaignBudget(cases=2, measurements=3)
        summary = run_gpu_campaign(device, "cuda13", budget=budget)

        assert summary["campaign"] == "XLAxiom bounded EPYC GPU campaign"
        assert summary["variant"] == "cuda13"
        assert summary["device"]["name"] == device.name
        assert summary["budget"]["cases"] == 2
        assert summary["budget"]["measurements"] == 3
        assert summary["budget"]["max_duration_s"] == 300.0
        assert summary["completion"]["truncated"] is False
        assert len(summary["cpu_vs_gpu_cases"]) == 2
        assert len(summary["gpu_eager_vs_jit_cases"]) == 2
        assert summary["pass_count"] == 4
        assert summary["anomalies"] == []
        assert len(summary["notes"]) == 3

    def test_run_gpu_campaign_detects_anomalies(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device()

        def mock_cpu_vs_gpu(seed, dev, var, measurements):
            classification = "pass" if seed == 2 else "numerical_discrepancy"
            return {
                "seed": seed,
                "numpy_vs_gpu_jit": {"classification": classification},
            }

        def mock_gpu_eager_vs_jit(seed, dev, var):
            return {
                "seed": seed,
                "classification": "pass",
            }

        monkeypatch.setattr("xlaxiom.gpu_campaign.cpu_vs_gpu", mock_cpu_vs_gpu)
        monkeypatch.setattr("xlaxiom.gpu_campaign.gpu_eager_vs_jit", mock_gpu_eager_vs_jit)

        budget = CampaignBudget(cases=2)
        summary = run_gpu_campaign(device, "cuda13", budget=budget)

        assert len(summary["anomalies"]) == 1
        anom = summary["anomalies"][0]
        assert anom["numpy_vs_gpu_jit"]["classification"] == "numerical_discrepancy"

    def test_run_gpu_campaign_timeout_guard(self, monkeypatch: pytest.MonkeyPatch) -> None:
        device = make_device()
        calls: list[str] = []

        times = [0.0, 10.0, 310.0]
        time_idx = 0

        def fake_perf_counter():
            nonlocal time_idx
            val = times[min(time_idx, len(times) - 1)]
            time_idx += 1
            return val

        def mock_cpu(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append("cpu")
            return {"numpy_vs_gpu_jit": {"classification": "pass"}}

        def mock_eager(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append("eager")
            return {"classification": "pass"}

        monkeypatch.setattr("time.perf_counter", fake_perf_counter)
        monkeypatch.setattr("xlaxiom.gpu_campaign.cpu_vs_gpu", mock_cpu)
        monkeypatch.setattr("xlaxiom.gpu_campaign.gpu_eager_vs_jit", mock_eager)

        budget = CampaignBudget(cases=5)
        summary = run_gpu_campaign(device, "cuda13", budget=budget)

        assert len(summary["cpu_vs_gpu_cases"]) == 1
        assert summary["gpu_eager_vs_jit_cases"] == []
        assert calls == ["cpu"]
        assert summary["completion"] == {
            "requested_cases_per_phase": 5,
            "cpu_vs_gpu_cases_completed": 1,
            "gpu_eager_vs_jit_cases_completed": 0,
            "duration_limit_reached": True,
            "truncated": True,
            "truncation_reason": "maximum_duration_reached",
        }

    def test_run_gpu_campaign_deadline_applies_to_eager_phase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        device = make_device()
        calls: list[str] = []
        times = [0.0, 1.0, 2.0, 301.0]
        time_idx = 0

        def fake_perf_counter():
            nonlocal time_idx
            value = times[min(time_idx, len(times) - 1)]
            time_idx += 1
            return value

        def mock_cpu_2(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append("cpu")
            return {"numpy_vs_gpu_jit": {"classification": "pass"}}

        def mock_eager_2(*args: object, **kwargs: object) -> dict[str, object]:
            calls.append("eager")
            return {"classification": "pass"}

        monkeypatch.setattr("time.perf_counter", fake_perf_counter)
        monkeypatch.setattr("xlaxiom.gpu_campaign.cpu_vs_gpu", mock_cpu_2)
        monkeypatch.setattr("xlaxiom.gpu_campaign.gpu_eager_vs_jit", mock_eager_2)

        summary = run_gpu_campaign(device, "cuda13", budget=CampaignBudget(cases=2))

        assert len(summary["cpu_vs_gpu_cases"]) == 2
        assert summary["gpu_eager_vs_jit_cases"] == []
        assert calls == ["cpu", "cpu"]
        assert summary["completion"]["truncated"] is True
        assert summary["completion"]["truncation_reason"] == "maximum_duration_reached"


class TestReplayJax40634:
    """Tests for replay_jax_40634_on_gpu."""

    def test_replay_jax_40634_on_gpu_without_gpu_raises_or_skips(self) -> None:
        try:
            import jax

            jax.devices("gpu")
        except RuntimeError:
            with pytest.raises(RuntimeError):
                replay_jax_40634_on_gpu()
            return
        except Exception:
            pytest.skip("GPU backend not configured")
            return

        result = replay_jax_40634_on_gpu()
        assert result["study"] == "JAX #40634 large-float32 gradient replay on GPU"
        assert len(result["observations"]) == 3

    def test_replay_jax_40634_on_gpu_mocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import jax
        import jax.numpy as jnp

        class FakeDevice:
            def __str__(self) -> str:
                return "gpu:0"

        monkeypatch.setattr(jax, "devices", lambda backend: [FakeDevice()])

        def fake_grad(func):
            def wrapper(val):
                class Blockable:
                    def block_until_ready(self):
                        raw = float(val)
                        if func == jnp.arcsinh:
                            return 1.0 / math.hypot(1.0, abs(raw))
                        return 1.0 / (math.sqrt(raw - 1.0) * math.sqrt(raw + 1.0))

                return Blockable()

            return wrapper

        def fake_jit(func):
            return func

        monkeypatch.setattr(jax, "grad", fake_grad)
        monkeypatch.setattr(jax, "jit", fake_jit)

        result = replay_jax_40634_on_gpu()
        assert result["study"] == "JAX #40634 large-float32 gradient replay on GPU"
        assert result["devices"] == ["gpu:0"]
        assert len(result["observations"]) == 3
        for obs in result["observations"]:
            assert obs["classification"] == "pass"
            assert obs["device"] == "gpu:0"
