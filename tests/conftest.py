import numpy as np
import pytest

from spare.synth.registry import FAMILIES


@pytest.fixture(scope="session")
def instances():
    rng = np.random.default_rng(1234)
    return {name: f.instance(rng, f"{name}_t0") for name, f in FAMILIES.items()}
