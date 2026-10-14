"""Verify the identity and required package data of built XLAxiom distributions.

This deliberately uses only the Python standard library so that it can run in the same
clean environment as the release build. It validates what will be uploaded, rather than
only the source tree that produced it.
"""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from xlaxiom import __version__

PACKAGE_NAME = "xlaxiom"
REQUIRED_WHEEL_MEMBERS = {
    "xlaxiom/py.typed",
    "xlaxiom/corpus/curated-v1.json",
    "xlaxiom/corpus/historical/jax-40634.json",
}
REQUIRED_SDIST_SUFFIXES = {
    "src/xlaxiom/py.typed",
    "src/xlaxiom/corpus/curated-v1.json",
    "src/xlaxiom/corpus/historical/jax-40634.json",
    "pyproject.toml",
}


def _metadata(raw: bytes, source: Path) -> tuple[str, str]:
    message = BytesParser().parsebytes(raw)
    name = message.get("Name")
    version = message.get("Version")
    if name is None or version is None:
        raise ValueError(f"{source} has incomplete core metadata")
    return name, version


def _single_artifact(directory: Path, pattern: str) -> Path:
    artifacts = sorted(directory.glob(pattern))
    if len(artifacts) != 1:
        raise ValueError(
            f"expected exactly one {pattern!r} distribution in {directory}, found {len(artifacts)}"
        )
    return artifacts[0]


def _verify_wheel(wheel: Path) -> tuple[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError(f"{wheel} must contain exactly one dist-info METADATA file")
        missing = REQUIRED_WHEEL_MEMBERS - names
        if missing:
            raise ValueError(f"{wheel} is missing required package data: {sorted(missing)}")
        return _metadata(archive.read(metadata_names[0]), wheel)


def _verify_sdist(sdist: Path) -> tuple[str, str]:
    with tarfile.open(sdist, "r:gz") as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]
        metadata_names = [name for name in names if name.endswith("/PKG-INFO")]
        if len(metadata_names) != 1:
            raise ValueError(f"{sdist} must contain exactly one PKG-INFO file")
        missing = [
            suffix
            for suffix in REQUIRED_SDIST_SUFFIXES
            if not any(name.endswith(f"/{suffix}") for name in names)
        ]
        if missing:
            raise ValueError(f"{sdist} is missing required source data: {sorted(missing)}")
        member = archive.getmember(metadata_names[0])
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError(f"could not read metadata from {sdist}")
        return _metadata(handle.read(), sdist)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--dist-dir", type=Path, default=Path("dist"))
    command.add_argument("--tag", help="optional release tag that must equal v<package version>")
    return command


def main() -> int:
    args = parser().parse_args()
    directory = args.dist_dir
    wheel = _single_artifact(directory, "*.whl")
    sdist = _single_artifact(directory, "*.tar.gz")
    wheel_metadata = _verify_wheel(wheel)
    sdist_metadata = _verify_sdist(sdist)

    expected = (PACKAGE_NAME, __version__)
    if wheel_metadata != expected or sdist_metadata != expected:
        raise SystemExit(
            "distribution metadata must match the package: "
            f"expected {expected!r}, wheel={wheel_metadata!r}, sdist={sdist_metadata!r}"
        )
    if args.tag is not None and args.tag != f"v{__version__}":
        raise SystemExit(f"release tag {args.tag!r} does not match package version {__version__!r}")

    print(f"verified {PACKAGE_NAME} {__version__}: {wheel.name}, {sdist.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
