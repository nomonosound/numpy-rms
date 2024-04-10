import math

import numpy as np
from numpy.typing import NDArray


def calculate_rms(a: NDArray):
    """Given a numpy array, return its RMS power level."""
    return np.sqrt(np.mean(np.square(a), axis=-1))


def rms_numpy(a: NDArray, window_size: int) -> NDArray:
    if 0 in a.shape:
        raise ValueError("Cannot input empty array")

    output_shape = list(a.shape)
    output_shape[-1] = math.floor(a.shape[-1] / window_size)
    output_array = np.zeros(shape=output_shape, dtype=a.dtype)

    output_i = 0
    for offset in range(0, a.shape[-1], window_size):
        rms = calculate_rms(a[..., offset : offset + window_size])
        output_array[output_i] = rms
        output_i += 1
    return output_array