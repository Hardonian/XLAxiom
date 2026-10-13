from unittest.mock import MagicMock, patch

import pytest

from xlaxiom.campaign import CampaignConfig, _run_isolated, run_cpu_campaign
from xlaxiom.comparison import Classification


def test_campaign_config_validation() -> None:
    # Valid default config should not raise
    config = CampaignConfig()
    config.validate()

    # Invalid cases
    with pytest.raises(ValueError, match="cases must be between 1 and 500"):
        CampaignConfig(cases=0).validate()
    with pytest.raises(ValueError, match="cases must be between 1 and 500"):
        CampaignConfig(cases=501).validate()

    # Invalid max_duration_s
    with pytest.raises(ValueError, match="max_duration_s must be between 1 and 900"):
        CampaignConfig(max_duration_s=0.0).validate()
    with pytest.raises(ValueError, match="max_duration_s must be between 1 and 900"):
        CampaignConfig(max_duration_s=901.0).validate()

    # Invalid case_timeout_s when isolate is True
    with pytest.raises(ValueError, match="case_timeout_s must be between 1 and 600"):
        CampaignConfig(isolate=True, case_timeout_s=0.0).validate()
    with pytest.raises(ValueError, match="case_timeout_s must be between 1 and 600"):
        CampaignConfig(isolate=True, case_timeout_s=601.0).validate()

    # When isolate is False, out-of-range case_timeout_s is ignored during validation
    CampaignConfig(isolate=False, case_timeout_s=0.0).validate()

    with pytest.raises(ValueError, match="case_memory_mb must be between 16 and 65536"):
        CampaignConfig(case_memory_mb=15).validate()


def test_run_isolated_success() -> None:
    config = CampaignConfig(isolate=True, case_timeout_s=10.0)
    mock_workload = MagicMock()
    mock_record = {"classification": "pass", "details": "ok"}
    mock_result = MagicMock()
    mock_result.status = "ok"
    mock_result.record = mock_record

    with patch("xlaxiom.safety.compare_isolated", return_value=mock_result) as mock_compare:
        record, failure = _run_isolated(mock_workload, "jax-jit", config)
        assert record == mock_record
        assert failure is None
        mock_compare.assert_called_once_with(mock_workload, "jax-jit", timeout_s=10.0)


def test_run_isolated_failure() -> None:
    config = CampaignConfig(isolate=True, case_timeout_s=10.0)
    mock_workload = MagicMock()
    mock_result = MagicMock()
    mock_result.status = "timeout"
    mock_result.error = "Execution timed out after 10s"

    with patch("xlaxiom.safety.compare_isolated", return_value=mock_result):
        record, failure = _run_isolated(mock_workload, "jax-jit", config)
        assert record is None
        assert failure == {"status": "timeout", "error": "Execution timed out after 10s"}


def test_run_isolated_passes_configured_memory_limit() -> None:
    config = CampaignConfig(isolate=True, case_timeout_s=10.0, case_memory_mb=512)
    mock_result = MagicMock(status="ok", record={"classification": "pass"})
    workload = MagicMock()

    with patch("xlaxiom.safety.compare_isolated", return_value=mock_result) as mock_compare:
        record, failure = _run_isolated(workload, "jax-jit", config)

    assert record == {"classification": "pass"}
    assert failure is None
    mock_compare.assert_called_once_with(workload, "jax-jit", timeout_s=10.0, memory_limit_mb=512)


def test_run_cpu_campaign_non_isolated_with_anomalies() -> None:
    config = CampaignConfig(cases=2, isolate=False)

    pass_comparison = MagicMock()
    pass_comparison.to_dict.return_value = {"classification": Classification.PASS.value}

    discrepancy_comparison = MagicMock()
    discrepancy_comparison.to_dict.return_value = {
        "classification": Classification.NUMERICAL_DISCREPANCY.value
    }

    # Alternate comparisons: pass then discrepancy
    side_effects = [pass_comparison, discrepancy_comparison, pass_comparison, pass_comparison]

    with patch("xlaxiom.campaign.run_comparison", side_effect=side_effects):
        result = run_cpu_campaign(config)

        assert result["campaign_version"] == "1.0"
        assert result["requested_cases"] == 2
        assert result["completed_cases"] == 2
        assert result["stop_reason"] == "completed"
        assert result["isolation"]["enabled"] is False
        assert result["isolation"]["case_timeout_s"] is None
        assert len(result["candidate_anomalies"]) == 1
        assert result["candidate_anomalies"][0]["seed"] == config.seed
        assert result["numpy_vs_jit_counts"][Classification.PASS.value] == 2
        assert result["eager_vs_jit_counts"][Classification.PASS.value] == 1
        assert result["eager_vs_jit_counts"][Classification.NUMERICAL_DISCREPANCY.value] == 1


def test_run_cpu_campaign_duration_budget_exhausted() -> None:
    config = CampaignConfig(cases=10, max_duration_s=1.0)

    call_count = 0

    def mock_perf_counter() -> float:
        nonlocal call_count
        call_count += 1
        # First call is start time = 0.0, second check in loop > 1.0
        if call_count == 1:
            return 0.0
        return 10.0

    with patch("time.perf_counter", side_effect=mock_perf_counter):
        result = run_cpu_campaign(config)
        assert result["completed_cases"] == 0
        assert result["stop_reason"] == "duration_budget_exhausted"


def test_run_cpu_campaign_isolated_mode() -> None:
    config = CampaignConfig(cases=1, isolate=True, case_timeout_s=30.0)

    failure_diag = {"status": "timeout", "error": "timed out"}

    with patch(
        "xlaxiom.campaign._run_isolated",
        side_effect=[(None, failure_diag), ({"classification": "pass"}, None)],
    ):
        result = run_cpu_campaign(config)
        assert result["completed_cases"] == 1
        assert result["isolation"]["enabled"] is True
        assert result["isolation"]["case_timeout_s"] == 30.0
        assert result["isolation"]["contained_count"] == 1
        assert len(result["contained_failures"]) == 1
        assert result["contained_failures"][0]["numpy_vs_jit"] == failure_diag
        assert result["contained_failures"][0]["eager_vs_jit"] is None
