"""Shared pytest fixtures for mattersim tests."""

import pytest
import torch


def _available_devices():
    """Return all available torch devices on this machine."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(
    params=_available_devices(),
    ids=lambda d: f"device={d}",
)
def device(request):
    """Yields each available device (cpu, cuda, mps)."""
    return request.param
