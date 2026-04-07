import pytest
import torch


def get_available_devices():
    """Return a list of available torch devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="Run tests only on the specified device (cpu, cuda, mps)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "device: mark test to run on torch devices")


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        requested = metafunc.config.getoption("--device")
        if requested:
            devices = [requested]
        else:
            devices = get_available_devices()
        metafunc.parametrize("device", devices, scope="session")
