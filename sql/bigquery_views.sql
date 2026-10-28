-- ============================================================================
-- XLAxiom Enterprise BigQuery Telemetry Views
-- Target: Google BigQuery Studio & Looker Studio Analytics
-- ============================================================================

-- 1. Daily Rolling Compiler Health & Regression Monitor
CREATE OR REPLACE VIEW \{PROJECT_ID}.xlaxiom_telemetry.v_daily_campaign_health\ AS
SELECT
    DATE(timestamp) AS campaign_date,
    environment.accelerator AS accelerator,
    environment.jax_version AS jax_version,
    COUNT(DISTINCT campaign_id) AS total_campaigns,
    SUM(total_cases) AS total_cases_evaluated,
    SUM(passed_cases) AS total_cases_passed,
    SUM(failed_cases) AS total_cases_failed,
    SUM(anomalies) AS total_anomalies_detected,
    ROUND(SAFE_DIVIDE(SUM(passed_cases), SUM(total_cases)) * 100.0, 2) AS pass_percentage,
    ROUND(AVG(duration_seconds), 2) AS avg_duration_seconds
FROM
    \{PROJECT_ID}.xlaxiom_telemetry.campaignsGROUP BY
    1, 2, 3
ORDER BY
    campaign_date DESC, accelerator;

-- 2. TPU Hardware Roofline & Efficiency Distribution
CREATE OR REPLACE VIEW \{PROJECT_ID}.xlaxiom_telemetry.v_tpu_roofline_efficiency\ AS
SELECT
    DATE(timestamp) AS benchmark_date,
    accelerator,
    regime,
    COUNT(DISTINCT workload_id) AS workload_count,
    ROUND(AVG(arithmetic_intensity), 2) AS avg_arithmetic_intensity_flops_per_byte,
    ROUND(AVG(achieved_tflops), 2) AS avg_achieved_tflops,
    ROUND(AVG(percent_peak_compute), 2) AS avg_percent_peak_compute,
    ROUND(MAX(percent_peak_compute), 2) AS max_percent_peak_compute,
    ROUND(AVG(measured_latency_ms), 3) AS avg_kernel_latency_ms
FROM
    \{PROJECT_ID}.xlaxiom_telemetry.rooflinesGROUP BY
    1, 2, 3
ORDER BY
    benchmark_date DESC, accelerator, avg_percent_peak_compute DESC;

-- 3. Metamorphic Invariant Sharpness & Compiler Discrepancy Heatmap
CREATE OR REPLACE VIEW \{PROJECT_ID}.xlaxiom_telemetry.v_metamorphic_sharpness\ AS
SELECT
    relation_name,
    backend,
    mode,
    COUNT(*) AS total_evaluations,
    COUNTIF(status != 'pass') AS discrepancy_count,
    ROUND(AVG(max_absolute_error), 8) AS avg_max_absolute_error,
    ROUND(MAX(max_absolute_error), 8) AS peak_max_absolute_error,
    ROUND(AVG(max_relative_error), 8) AS avg_max_relative_error,
    ROUND(AVG(sharpness_ratio), 4) AS avg_mutation_sharpness_ratio
FROM
    \{PROJECT_ID}.xlaxiom_telemetry.propertiesGROUP BY
    1, 2, 3
ORDER BY
    discrepancy_count DESC, avg_max_absolute_error DESC;
