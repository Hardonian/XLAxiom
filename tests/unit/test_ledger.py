from pathlib import Path

from xlaxiom.ledger import completed_shard, load_ledger, save_shard


def test_campaign_ledger_resumes_only_the_exact_shard(tmp_path: Path) -> None:
    path = tmp_path / "gpu-ledger.json"
    identity = {"device": "redacted", "variant": "cuda12", "measurements": 3}
    shard = {"index": 0, "count": 2, "seed": 4, "cases": 2}
    save_shard(path, identity, shard, {"pass_count": 4})

    loaded = load_ledger(path)

    assert completed_shard(loaded, identity, shard) == {"pass_count": 4}
    assert completed_shard(loaded, identity, {**shard, "index": 1}) is None
