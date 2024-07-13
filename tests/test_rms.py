import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import numpy_rms
from numpy_rms.fallback import rms_numpy


def test_rms_large_array(benchmark):
    arr = np.arange(100_000_000, dtype=np.float32)
    rms = benchmark(numpy_rms.rms, arr, window_size=5000)
    assert rms.shape == (20_000,)
    assert rms[0] == pytest.approx(2886.3184)
    assert rms[1] == pytest.approx(7637.1353)


def test_rms_2d_mono():
    arr = np.ones(shape=(1, 500), dtype=np.float32)
    rms = numpy_rms.rms(arr, window_size=50)
    rms_numpy_fallback = rms_numpy(arr, window_size=50)
    assert rms.shape == (1, 10)
    assert rms_numpy_fallback.shape == (1, 10)
    assert rms[0, 0] == 1.0
    assert_array_almost_equal(rms, rms_numpy_fallback)


def test_rms_2d_stereo():
    arr = np.ones(shape=(2, 500), dtype=np.float32)
    arr[1, :] = 2.0
    rms = numpy_rms.rms(arr, window_size=50)
    rms_numpy_fallback = rms_numpy(arr, window_size=50)
    assert rms.shape == (2, 10)
    assert rms_numpy_fallback.shape == (2, 10)
    assert rms[0, 0] == 1.0
    assert rms[1, 1] == 2.0
    assert_array_almost_equal(rms, rms_numpy_fallback)


def test_rms_numpy_fallback_large_array(benchmark):
    arr = np.arange(100_000_000, dtype=np.float32)
    rms = benchmark(rms_numpy, arr, window_size=5000)
    assert rms.shape == (20_000,)
    assert rms[0] == pytest.approx(2886.3184)
    assert rms[1] == pytest.approx(7637.1353)


def test_not_divisible_by_window_size():
    arr = np.arange(155, dtype=np.float32)
    rms = numpy_rms.rms(arr, window_size=16)
    rms_numpy_fallback = rms_numpy(arr, window_size=16)
    assert rms.shape == rms_numpy_fallback.shape
    assert_array_almost_equal(rms, rms_numpy_fallback)


def test_rms_window_size_not_divisible_by_8():
    arr = np.arange(130, dtype=np.float32)
    rms = numpy_rms.rms(arr, window_size=13)
    rms_numpy_fallback = rms_numpy(arr, window_size=13)
    assert rms.shape == rms_numpy_fallback.shape
    assert_array_almost_equal(rms, rms_numpy_fallback)


def test_rms_window_size_smaller_than_8():
    arr = np.arange(40, dtype=np.float32)
    rms = numpy_rms.rms(arr, window_size=4)
    rms_numpy_fallback = rms_numpy(arr, window_size=4)
    assert rms.shape == rms_numpy_fallback.shape
    assert_array_almost_equal(rms, rms_numpy_fallback)


def test_rms_window_size_none():
    arr = np.arange(40, dtype=np.float32)
    rms = numpy_rms.rms(arr)
    assert rms.shape == (1,)
