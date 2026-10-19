import warnings

import pytest

from continuous_brain_snn.config import small_config
from continuous_brain_snn.engine import SNNEngine

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def cfg():
    return small_config(seed=0)


@pytest.fixture
def engine(cfg):
    return SNNEngine(cfg)
