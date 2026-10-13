from xlaxiom.storage import atomic_write_text, save_json


def test_atomic_write_replaces_existing_content_without_temp_files(tmp_path) -> None:
    target = tmp_path / "evidence.json"
    target.write_text("old", encoding="utf-8")

    saved = atomic_write_text(target, "new")

    assert saved == target
    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_json_creates_parent_directory_atomically(tmp_path) -> None:
    target = tmp_path / "nested" / "evidence.json"

    saved = save_json(target, {"record": "complete"})

    assert saved == target
    assert target.read_text(encoding="utf-8") == '{\n  "record": "complete"\n}\n'
