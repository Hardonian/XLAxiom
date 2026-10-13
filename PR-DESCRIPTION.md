# 🧪 [testing improvement] Add unit test suite for gpu_campaign.py

### 🎯 What: The testing gap addressed
Added a comprehensive test suite in `tests/unit/test_gpu_campaign.py` targeting `src/xlaxiom/gpu_campaign.py`, which previously had no dedicated unit tests.

### 📊 Coverage: What scenarios are now tested
- **`CampaignBudget` validation:** Enforces bounds on `cases` (1..24) and `measurements` (1..20).
- **`campaign_cases` generation:** Verifies workload family rotation and seed derivation for various budgets.
- **`cpu_vs_gpu` comparisons:** Verifies schema structure, timing records, and graceful handling of missing CPU baselines under isolated CUDA runtimes.
- **`gpu_eager_vs_jit` observations:** Validates comparison dictionary schema and explicit device index placement.
- **`run_gpu_campaign` entrypoint:** Tests full campaign workflow, anomaly detection across cases, default/custom budget overrides, and max duration time limit guards.
- **`replay_jax_40634_on_gpu` study:** Tests hardware execution / fallback behavior and mocked JAX gradient calculations for the JAX #40634 issue replay.

### ✨ Result: The improvement in test coverage
- Added 16 new unit tests covering all functions and edge cases in `src/xlaxiom/gpu_campaign.py`.
- Ensured deterministic execution using `pytest.MonkeyPatch` doubles without needing a live GPU host.
