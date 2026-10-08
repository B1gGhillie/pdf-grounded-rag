"""Test selection rules for local development and CI."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests marked gpu (requires a usable CUDA PyTorch installation)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="pass --run-gpu to run CUDA-dependent tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
