"""Upstream JAX standalone reproduction runner.

Executed by maintainers or GitHub Actions to verify an XLAxiom bug report.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_reproduction(script_path: str) -> int:
    print(f"[upstream-jax-verifier] Executing standalone repro: {script_path}")
    proc = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    print("STDOUT:\n", proc.stdout)
    if proc.stderr:
        print("STDERR:\n", proc.stderr)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify standalone JAX bug report")
    parser.add_argument("script", help="Path to standalone reproduction Python script")
    args = parser.parse_args()
    return run_reproduction(args.script)


if __name__ == "__main__":
    sys.exit(main())
