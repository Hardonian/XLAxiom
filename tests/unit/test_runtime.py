import pytest

from xlaxiom.runtime import runtime_profile


def test_runtime_profiles_are_explicit_and_bounded() -> None:
    assert runtime_profile("single-thread").environment["OMP_NUM_THREADS"] == "1"
    assert runtime_profile("four-thread").environment["MKL_NUM_THREADS"] == "4"
    with pytest.raises(ValueError, match="unknown runtime profile"):
        runtime_profile("all-cores")
