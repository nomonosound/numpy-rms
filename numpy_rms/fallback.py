import numpy as np
from numpy.typing import NDArray


def calculate_rms(a: NDArray):
    """Given a numpy array, return its RMS power level."""
    return np.sqrt(np.mean(np.square(a), axis=-1))


def rms_numpy(a: NDArray, window_size: int, rms_series_output: NDArray) -> NDArray:
    output_i = 0
    for offset in range(0, a.shape[-1], window_size):
        rms = calculate_rms(a[..., offset : offset + window_size])
        rms_series_output[output_i] = rms
        output_i += 1
    return rms_series_output
