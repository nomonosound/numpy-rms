import numpy as np
import pytest

import numpy_rms
from numpy_rms.fallback import rms_numpy


def test_rms(benchmark):
    arr = np.arange(100_000_000, dtype=np.float32)
    rms = benchmark(numpy_rms.rms, arr, window_size=5000)
    assert rms.shape == (20_000,)
    assert rms[0] == pytest.approx(2886.3184)
    assert rms[1] == pytest.approx(7637.1353)

def test_rms_numpy_fallback(benchmark):
    arr = np.arange(100_000_000, dtype=np.float32)
    rms = benchmark(rms_numpy, arr, window_size=5000)
    assert rms.shape == (20_000,)
    assert rms[0] == pytest.approx(2886.3184)
    assert rms[1] == pytest.approx(7637.1353)
