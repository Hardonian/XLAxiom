"""Validation tests for on-boarded external ecosystem assets.

Verifies multi-generation TPU evidence suites, upstream integration PR packages,
Terraform infrastructure-as-code, and Google Cloud Marketplace solution specs.
"""

from __future__ import annotations

import json
from pathlib import Path


def test_multigen_tpu_evidence_manifests() -> None:
    root = Path(__file__).resolve().parents[2]
    generations = ("tpu-v4", "tpu-v5p", "tpu-v6e")

    for gen in generations:
        gen_dir = root / "evidence" / gen
        assert gen_dir.exists(), f"Evidence directory missing: {gen_dir}"

        manifest_path = gen_dir / "manifest.json"
        assert manifest_path.exists(), f"Manifest missing: {manifest_path}"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema_version"] == "offline-tpu-qualification-v1"
        assert manifest["passed"] is True
        assert manifest["mode"] == "simulated"
        assert len(manifest["limitations"]) >= 1

        # Verify all declared artifacts exist on disk
        for artifact_name, meta in manifest["artifacts"].items():
            artifact_file = gen_dir / artifact_name
            assert artifact_file.exists(), f"Artifact {artifact_name} missing in {gen}"
            assert artifact_file.stat().st_size == meta["bytes"]

        # Verify topology specification
        topo_path = gen_dir / "hardware-topology.json"
        assert topo_path.exists()
        topo_data = json.loads(topo_path.read_text(encoding="utf-8"))
        assert "topology" in topo_data
        assert "chips" in topo_data


def test_upstream_ecosystem_packages() -> None:
    root = Path(__file__).resolve().parents[2]

    # JAX package
    jax_dir = root / "upstream" / "jax"
    assert (jax_dir / "PULL_REQUEST.md").exists()
    assert (jax_dir / "bug_report_patch.yml").exists()
    assert (jax_dir / "reproduce_report.py").exists()

    # OpenXLA package
    openxla_dir = root / "upstream" / "openxla"
    assert (openxla_dir / "RFC_COMPILER_PASS_ISOLATION.md").exists()
    assert (openxla_dir / "stablehlo_opt_runner.py").exists()

    # MaxText package
    maxtext_dir = root / "upstream" / "maxtext"
    assert (maxtext_dir / "test_maxtext_numerical_parity.py").exists()


def test_terraform_infrastructure_specs() -> None:
    root = Path(__file__).resolve().parents[2]
    tf_dir = root / "terraform"

    main_tf = (tf_dir / "main.tf").read_text(encoding="utf-8")
    assert "google_kms_crypto_key" in main_tf
    assert "google_bigquery_dataset" in main_tf
    assert "google_storage_bucket" in main_tf
    assert "google_service_account" in main_tf

    vars_tf = (tf_dir / "variables.tf").read_text(encoding="utf-8")
    assert "project_id" in vars_tf

    outputs_tf = (tf_dir / "outputs.tf").read_text(encoding="utf-8")
    assert "kms_key_id" in outputs_tf


def test_enterprise_bootstrap_scripts() -> None:
    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"

    sh_script = (scripts_dir / "enterprise_gcp_bootstrap.sh").read_text(encoding="utf-8")
    assert "xlaxiom-keyring" in sh_script
    assert "xlaxiom_telemetry" in sh_script

    ps_script = (scripts_dir / "enterprise_gcp_bootstrap.ps1").read_text(encoding="utf-8")
    assert "xlaxiom-keyring" in ps_script
    assert "xlaxiom_telemetry" in ps_script

    tpu_sh = (scripts_dir / "tpu" / "provision_and_collect_all.sh").read_text(encoding="utf-8")
    assert "v4-8" in tpu_sh
    assert "v6e-4" in tpu_sh

    tpu_ps = (scripts_dir / "tpu" / "provision_and_collect_all.ps1").read_text(encoding="utf-8")
    assert "v4-8" in tpu_ps
    assert "v6e-4" in tpu_ps


def test_marketplace_and_publish_manifests() -> None:
    root = Path(__file__).resolve().parents[2]

    # Marketplace solution
    sol_def = (root / "marketplace" / "solution_definition.yaml").read_text(encoding="utf-8")
    assert "xlaxiom-compiler-audit" in sol_def
    assert "2.2.0" in sol_def

    # Workbench extension
    ext_path = root / "marketplace" / "vertex_workbench_extension.json"
    assert ext_path.exists()
    ext_data = json.loads(ext_path.read_text(encoding="utf-8"))
    assert ext_data["name"] == "xlaxiom-workbench-extension"

    # PyPI and SLSA GitHub workflow
    pub_path = root / ".github" / "workflows" / "publish.yml"
    assert pub_path.exists()
    pub_text = pub_path.read_text(encoding="utf-8")
    assert "gh-action-pypi-publish" in pub_text
    assert "attest-build-provenance" in pub_text
