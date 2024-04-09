import math

import _numpy_rms
import numpy as np
from numpy.typing import NDArray

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def rms(a: NDArray, window_size: int) -> NDArray:
    if 0 in a.shape:
        raise ValueError("Cannot input empty array")

    output_shape = list(a.shape)
    output_shape[-1] = math.floor(a.shape[-1] / window_size)
    output_array = np.zeros(shape=output_shape, dtype=a.dtype)

    if a.dtype == np.dtype("float32"):
        if a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]:
            _numpy_rms.lib.rms(
                _numpy_rms.ffi.cast("float *", a.ctypes.data),
                a.size,
                window_size,
                _numpy_rms.ffi.cast("float *", output_array.ctypes.data),
                output_array.size,
            )
            return output_array
        if a.ndim == 1:
            raise Exception("Not implemented yet")
    else:
        raise Exception("Not implemented yet")
    return output_array
