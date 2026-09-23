import warnings

import pytest
import torch

from continuous_brain.config import BrainConfig
from continuous_brain.core import ReferenceLIFConfig, ReferenceLIFCore
from continuous_brain.testing import TinyBackbone

warnings.filterwarnings("ignore", category=UserWarning)


def small_core(batch_size: int = 1, seed: int = 0) -> ReferenceLIFCore:
    return ReferenceLIFCore(ReferenceLIFConfig(n_neurons=256, n_input=32, n_projection=64, seed=seed),
                            batch_size=batch_size)


def small_config() -> BrainConfig:
    cfg = BrainConfig()
    cfg.transducer.n_populations = 16
    cfg.transducer.n_latent_tokens = 8
    cfg.transducer.latent_dim = 64
    cfg.transducer.n_heads = 4
    cfg.bridge.prefix_mlp_hidden = 128
    cfg.bridge.writeback_hidden = 64
    cfg.loop.max_new_tokens = 8
    cfg.sleep.bridge_steps = 20
    cfg.sleep.bridge_lr = 3e-3
    cfg.gate.threshold = 10.0   # only messages trigger consultations in tests
    return cfg


@pytest.fixture
def core():
    return small_core()


@pytest.fixture
def backbone():
    return TinyBackbone(hidden=64)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
