import subprocess
import sys


def test_fresh_process_reproduction(tmp_path) -> None:
    original = tmp_path / "original.json"
    reduced = tmp_path / "reduced.json"
    generate = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "generate",
            "--synthetic",
            "--seed",
            "4",
            "--output",
            str(original),
        ],
        capture_output=True,
        text=True,
    )
    assert generate.returncode == 0, generate.stderr
    minimized = subprocess.run(
        [
            sys.executable,
            "-m",
            "xlaxiom",
            "minimize",
            "--case",
            str(original),
            "--output",
            str(reduced),
        ],
        capture_output=True,
        text=True,
    )
    assert minimized.returncode == 0, minimized.stderr
    reproduced = subprocess.run(
        [sys.executable, "-m", "xlaxiom", "reproduce", "--case", str(reduced)],
        capture_output=True,
        text=True,
    )
    assert reproduced.returncode == 0, reproduced.stdout
