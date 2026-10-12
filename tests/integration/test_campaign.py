from xlaxiom.campaign import CampaignConfig, run_cpu_campaign


def test_bounded_cpu_campaign_compares_all_execution_modes() -> None:
    record = run_cpu_campaign(CampaignConfig(cases=4, max_duration_s=60))
    assert record["completed_cases"] == 4
    assert record["numpy_vs_jit_counts"]["pass"] == 4
    assert record["eager_vs_jit_counts"]["pass"] == 4
