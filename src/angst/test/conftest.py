import pytest
import numpy as np

# seed for random number generator
RNG_SEED = None


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)
