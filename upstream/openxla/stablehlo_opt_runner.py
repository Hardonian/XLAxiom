"""OpenXLA StableHLO-Opt subprocess pass isolation harness.

Pipes MLIR StableHLO text through individual compiler passes and checks parity.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_stablehlo_pass(
    mlir_text: str, pass_name: str, opt_binary: str = "stablehlo-opt"
) -> tuple[int, str]:
    """Execute a single StableHLO optimization pass using the official C++ binary."""
    cmd = [opt_binary, f"--{pass_name}"]
    try:
        proc = subprocess.run(cmd, input=mlir_text, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout if proc.returncode == 0 else proc.stderr
    except FileNotFoundError:
        return 1, f"Binary '{opt_binary}' not found on PATH. Install OpenXLA tools."


def main() -> int:
    parser = argparse.ArgumentParser(description="Run StableHLO pass isolation")
    parser.add_argument("--mlir", required=True, help="Path to MLIR text file")
    parser.add_argument("--pass-name", default="canonicalize", help="MLIR pass to execute")
    parser.add_argument("--opt-bin", default="stablehlo-opt", help="Path to stablehlo-opt")
    args = parser.parse_args()

    with open(args.mlir, encoding="utf-8") as f:
        text = f.read()

    rc, out = run_stablehlo_pass(text, args.pass_name, args.opt_bin)
    print(out)
    return rc


if __name__ == "__main__":
    sys.exit(main())
