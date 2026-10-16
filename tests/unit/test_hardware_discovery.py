from xlaxiom.hardware import (
    discover_apple_silicon,
    discover_rocm_devices,
    hardware_report,
)


def test_discover_apple_silicon() -> None:
    result = discover_apple_silicon()
    assert isinstance(result, dict)
    assert "is_apple_silicon" in result
    assert "platform" in result
    assert "metal_supported" in result
    assert "mps_available" in result


def test_discover_rocm_devices() -> None:
    result = discover_rocm_devices()
    assert isinstance(result, list)


def test_hardware_report_includes_apple_and_rocm() -> None:
    report = hardware_report()
    assert "apple_silicon" in report
    assert "rocm_devices" in report
    assert "devices" in report
