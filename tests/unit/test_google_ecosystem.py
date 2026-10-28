"""End-to-end unit and integration tests for the Google Ecosystem product readiness suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from xlaxiom.bigquery import (
    CAMPAIGN_TABLE_SCHEMA,
    PROPERTY_TABLE_SCHEMA,
    ROOFLINE_TABLE_SCHEMA,
    export_to_bigquery,
    format_campaign_rows,
)
from xlaxiom.bisect import BisectResult, BisectStep, query_gcs_nightly_wheels
from xlaxiom.cli import main
from xlaxiom.cloud.storage import (
    clear_mock_gcs,
    is_gcs_uri,
    parse_gcs_uri,
    read_gcs_text,
    write_gcs_text,
)
from xlaxiom.cloud.tpu import (
    CloudTPUManager,
    MultiHostTPUOrchestrator,
)
from xlaxiom.cloud.vertex import create_vertex_job_spec, generate_kfp_component_yaml
from xlaxiom.cloud.xpk import generate_xpk_workload_spec
from xlaxiom.colab import colab_badge_markdown, generate_colab_notebook, save_colab_notebook
from xlaxiom.comparison import Classification, ComparisonResult
from xlaxiom.flax import (
    build_synthetic_attention_block,
    build_synthetic_rmsnorm_block,
    build_synthetic_swiglu_block,
)
from xlaxiom.generation import GenerationConfig, generate
from xlaxiom.kms import generate_slsa_provenance, sign_kms_attestation, verify_kms_attestation
from xlaxiom.model import ComparisonPolicy
from xlaxiom.pipeline import run_backend, run_comparison
from xlaxiom.stablehlo_interpreter import interpret_stablehlo
from xlaxiom.storage import (
    CaseArtifact,
    load_case,
    save_case,
)
from xlaxiom.telemetry import format_cloud_log, trace_compiler_span
from xlaxiom.triage import format_triage_issue
from xlaxiom.xla_passes import isolate_xla_compiler_pass


def test_gcs_storage_layer() -> None:
    clear_mock_gcs()
    uri = "gs://my-bucket/cases/test.json"
    assert is_gcs_uri(uri)
    assert not is_gcs_uri("local/path.json")

    b, blob = parse_gcs_uri(uri)
    assert b == "my-bucket"
    assert blob == "cases/test.json"

    with pytest.raises(ValueError, match="invalid GCS URI format"):
        parse_gcs_uri("invalid-uri")

    content = '{"hello": "world"}'
    write_gcs_text(uri, content)
    read_back = read_gcs_text(uri)
    assert read_back == content


def test_gcs_case_artifact_roundtrip() -> None:
    clear_mock_gcs()
    workload = generate(GenerationConfig(seed=42))
    artifact = CaseArtifact(workload=workload, provenance="gcs-test")

    uri = "gs://evidence-bucket/runs/case_001.json"
    save_case(uri, artifact)

    loaded = load_case(uri)
    assert loaded.workload.digest() == artifact.workload.digest()
    assert loaded.provenance == "gcs-test"


def test_cloud_tpu_manager_and_multihost() -> None:
    mgr = CloudTPUManager(
        project_id="test-proj",
        zone="us-south1-a",
        accelerator_type="v5litepod-4",
    )
    spec = mgr.get_spec()
    assert spec.generation == "v5e"
    assert spec.topology_type == "2D Mesh"
    assert spec.peak_tflops_bf16 == 197.0

    result = mgr.run_evidence_suite(cases=10, dry_run=True)
    assert result.passed
    assert result.stages_executed == 7
    assert "tpu-job-" in result.job_id
    html = result._repr_html_()
    assert "Cloud TPU Execution" in html

    # Multi-host coordinator
    orchestrator = MultiHostTPUOrchestrator(pod_slice_type="v5e-16", num_hosts=4, chips_per_host=4)
    assert orchestrator.total_chips == 16
    env_vars = orchestrator.generate_worker_environment(host_index=1, coordinator_ip="10.0.0.1")
    assert env_vars["JAX_PROCESS_ID"] == "1"
    assert env_vars["JAX_NUM_PROCESSES"] == "4"

    sim_reduction = orchestrator.simulate_distributed_reduction(tensor_size_mb=64.0)
    assert sim_reduction["ring_steps"] == 30
    assert sim_reduction["simulated_latency_ms"] > 0


def test_cloud_vertex_and_xpk() -> None:
    vspec = create_vertex_job_spec(display_name="audit-run", accelerator="TPU_V5E")
    vdict = vspec.to_dict()
    assert vdict["displayName"] == "audit-run"
    assert "jobSpec" in vdict

    kfp_yaml = generate_kfp_component_yaml()
    assert "xlaxiom_compiler_verification" in kfp_yaml
    assert "case_gcs_uri" in kfp_yaml

    xspec = generate_xpk_workload_spec(workload_name="eval-1", cluster_name="tpu-c1")
    cli_cmd = xspec.to_cli_command()
    assert "xpk workload create" in cli_cmd
    assert "--workload=eval-1" in cli_cmd


def test_stablehlo_reference_interpreter() -> None:
    workload = generate(GenerationConfig(seed=12, max_nodes=5))
    res = interpret_stablehlo(workload)
    assert res.status.value == "ok"
    assert res.backend == "stablehlo-interpreter"
    assert res.outputs is not None
    assert len(res.outputs) == len(workload.outputs)

    # Pipeline integration
    pipe_res = run_backend(workload, backend="stablehlo-interpreter")
    assert pipe_res.status.value == "ok"

    comp_res = run_comparison(workload, mode="stablehlo-interpreter")
    assert comp_res.classification.value in ("pass", "numerical_discrepancy")


def test_xla_compiler_pass_isolation() -> None:
    workload = generate(GenerationConfig(seed=15, max_nodes=4))
    res = isolate_xla_compiler_pass(workload)
    assert res.workload_digest == workload.digest()
    assert res.total_passes == 9
    assert len(res.steps) == 9

    html = res._repr_html_()
    assert "XLA Pass Isolation" in html

    # With fault injected pass
    res_fault = isolate_xla_compiler_pass(workload, fault_inject_pass="fusion")
    assert res_fault.divergent_pass is not None
    assert "fusion" in res_fault.divergent_pass


def test_flax_and_maxtext_architectures() -> None:
    attn = build_synthetic_attention_block(seq_len=8, d_model=32, num_heads=2)
    attn.validate()
    assert len(attn.nodes) == 10
    assert attn.digest() != ""

    rmsnorm = build_synthetic_rmsnorm_block(seq_len=8, d_model=32)
    rmsnorm.validate()
    assert len(rmsnorm.nodes) == 5

    swiglu = build_synthetic_swiglu_block(seq_len=8, d_model=32, d_ff=64)
    swiglu.validate()
    assert len(swiglu.nodes) == 7


def test_colab_notebook_generator(tmp_path: Path) -> None:
    workload = generate(GenerationConfig(seed=7))
    case = CaseArtifact(workload=workload)

    notebook = generate_colab_notebook(case, title="Colab Test", accelerator="TPU")
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 4

    out_file = tmp_path / "notebook.ipynb"
    save_colab_notebook(notebook, out_file)
    assert out_file.exists()

    badge = colab_badge_markdown("Hardonian/XLAxiom", "notebooks/test.ipynb")
    assert "colab-badge.svg" in badge
    assert "colab.research.google.com" in badge


def test_kms_attestation_and_slsa(tmp_path: Path) -> None:
    subject = tmp_path / "evidence.json"
    subject.write_text('{"verdict": "pass"}', encoding="utf-8")

    key_name = "projects/test/locations/global/keyRings/r1/cryptoKeys/k1/cryptoKeyVersions/1"
    att = sign_kms_attestation(subject, key_name, mock_mode=True)
    assert att["kms_key_resource"] == key_name
    assert "signature_base64" in att

    valid = verify_kms_attestation(att, key_name)
    assert valid

    invalid = verify_kms_attestation(att, "wrong-key")
    assert not invalid

    slsa = generate_slsa_provenance("evidence.json", "sha256digest123")
    assert slsa["_type"] == "https://in-toto.io/Statement/v1"
    assert slsa["predicateType"] == "https://slsa.dev/provenance/v1"


def test_bigquery_and_telemetry() -> None:
    camp_dict = {
        "campaign_id": "c-001",
        "backend": "jax-tpu-jit",
        "accelerator": "v5litepod-4",
        "cases": [
            {
                "seed": 10,
                "workload_digest": "abc",
                "status": "pass",
                "max_absolute_error": 1e-6,
                "max_relative_error": 1e-5,
                "duration_ms": 1.5,
            }
        ],
    }
    rows = format_campaign_rows(camp_dict)
    assert len(rows) == 1
    assert rows[0]["campaign_id"] == "c-001"
    assert rows[0]["passed"] is True

    exported = export_to_bigquery(rows, "test.table")
    assert exported == 1

    assert len(CAMPAIGN_TABLE_SCHEMA) > 0
    assert len(PROPERTY_TABLE_SCHEMA) > 0
    assert len(ROOFLINE_TABLE_SCHEMA) > 0

    log = format_cloud_log("test message", severity="INFO")
    assert log["severity"] == "INFO"
    assert log["labels"]["tool"] == "xlaxiom"

    with trace_compiler_span("compile", "digest123") as span:
        pass
    assert span["total_ms"] >= 0.0


def test_repr_html_displays() -> None:
    workload = generate(GenerationConfig(seed=21))
    case = CaseArtifact(workload=workload)
    assert "XLAxiom Case Artifact" in case._repr_html_()

    comp = ComparisonResult(
        classification=Classification.PASS,
        explanation="Numerical agreement verified",
        policy=ComparisonPolicy(),
    )
    assert "Comparison Verdict" in comp._repr_html_()

    b_step = BisectStep(
        version="0.5.1",
        index=0,
        status="ok",
        classification="pass",
        duration_s=1.2,
    )
    b_res = BisectResult(
        case_path="case.json",
        total_versions_considered=5,
        steps_executed=1,
        tested_steps=(b_step,),
        first_bad_version=None,
        last_good_version="0.5.1",
        reproduction_command="xlaxiom bisect ...",
    )
    assert "Compiler Version Bisection Verdict" in b_res._repr_html_()


def test_query_gcs_nightly_wheels() -> None:
    wheels = query_gcs_nightly_wheels(timeout_s=0.5)
    assert isinstance(wheels, list)
    assert len(wheels) > 0


def test_triage_buganizer_target(tmp_path: Path) -> None:
    workload = generate(GenerationConfig(seed=30))
    case_path = tmp_path / "case.json"
    save_case(case_path, CaseArtifact(workload=workload))

    triage_md = format_triage_issue(case_path, target_tracker="buganizer")
    assert "Google Internal Issue (Buganizer)" in triage_md
    assert "Open In Colab" in triage_md


def test_cli_google_ecosystem_commands(tmp_path: Path) -> None:
    workload = generate(GenerationConfig(seed=50))
    case_path = tmp_path / "case.json"
    save_case(case_path, CaseArtifact(workload=workload))

    # Colab
    nb_path = tmp_path / "out.ipynb"
    ret = main(["colab", "--case", str(case_path), "--output", str(nb_path)])
    assert ret == 0
    assert nb_path.exists()

    # Cloud TPU status
    ret = main(["cloud", "tpu-status", "--accelerator", "v5litepod-4"])
    assert ret == 0

    # Cloud TPU run (dry run)
    manifest_path = tmp_path / "manifest.json"
    ret = main(["cloud", "tpu-run", "--cases", "5", "--dry-run", "--output", str(manifest_path)])
    assert ret == 0
    assert manifest_path.exists()

    # Cloud Vertex spec
    spec_path = tmp_path / "vertex.json"
    ret = main(["cloud", "vertex-spec", "--output", str(spec_path)])
    assert ret == 0
    assert spec_path.exists()

    # Cloud XPK spec
    ret = main(["cloud", "xpk-spec", "--workload", "test-w", "--cluster", "c1"])
    assert ret == 0

    # XLA pass-isolate
    pass_path = tmp_path / "pass.json"
    ret = main(["xla", "pass-isolate", "--case", str(case_path), "--output", str(pass_path)])
    assert ret == 0
    assert pass_path.exists()

    # BigQuery schema
    ret = main(["bigquery", "schema", "--table", "campaign"])
    assert ret == 0

    # KMS sign & verify
    key = "projects/p/locations/l/keyRings/r/cryptoKeys/k/cryptoKeyVersions/1"
    att_path = tmp_path / "att.json"
    ret = main(
        [
            "kms",
            "sign",
            "--subject",
            str(case_path),
            "--key",
            key,
            "--mock",
            "--slsa",
            "--output",
            str(att_path),
        ]
    )
    assert ret == 0
    assert att_path.exists()

    ret = main(["kms", "verify", "--attestation", str(att_path), "--key", key])
    assert ret == 0


def test_low_precision_dtypes() -> None:
    from xlaxiom.model import DTYPES, InputSpec, Node, Workload, materialize_inputs

    assert "float16" in DTYPES
    target_dtype = "bfloat16" if "bfloat16" in DTYPES else "float16"

    workload = Workload(
        identifier="low-prec-test",
        seed=123,
        inputs=(
            InputSpec("x", (4, 4), dtype=target_dtype, value_mode="random"),
            InputSpec("y", (4, 4), dtype=target_dtype, value_mode="finite_edges"),
        ),
        nodes=(Node("add", ("x", "y"), "out"),),
        outputs=("out",),
    )
    workload.validate()
    inputs = materialize_inputs(workload)
    assert inputs["x"].shape == (4, 4)
    assert str(inputs["x"].dtype) == target_dtype
    assert inputs["y"].shape == (4, 4)


def test_flax_nnx_tracing() -> None:
    from typing import Any

    import numpy as np

    from xlaxiom.flax import trace_flax_nnx_module

    class DummyNNXModule:
        def __call__(self, x: Any) -> Any:
            import jax.numpy as jnp

            return jnp.sin(x)

    module = DummyNNXModule()
    x = np.ones((4, 4), dtype=np.float32)
    workload = trace_flax_nnx_module(module, x, identifier="nnx-test")
    assert workload.identifier == "nnx-test"
    assert len(workload.nodes) >= 1

    with pytest.raises(TypeError, match="expected callable"):
        trace_flax_nnx_module("not_a_module", x)  # type: ignore[arg-type]


def test_colab_examples_and_bigquery_views() -> None:
    import json
    from pathlib import Path

    colab_dir = Path("examples/colab")
    assert colab_dir.exists()
    notebooks = list(colab_dir.glob("*.ipynb"))
    assert len(notebooks) >= 3

    for nb_path in notebooks:
        with open(nb_path, encoding="utf-8") as f:
            data = json.load(f)
            assert data["nbformat"] == 4
            assert len(data["cells"]) >= 3

    sql_path = Path("sql/bigquery_views.sql")
    assert sql_path.exists()
    sql_text = sql_path.read_text(encoding="utf-8")
    assert "v_daily_campaign_health" in sql_text
    assert "v_tpu_roofline_efficiency" in sql_text
    assert "v_metamorphic_sharpness" in sql_text
