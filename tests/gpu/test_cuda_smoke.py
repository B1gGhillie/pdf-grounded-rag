"""Opt-in CUDA health check; no network or model weights are required."""

import pytest


pytestmark = pytest.mark.gpu


def test_cuda_tensor_round_trip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to PyTorch")

    values = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    assert values.square().sum().item() == 14.0
