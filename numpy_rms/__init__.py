from typing import Optional

import _numpy_rms
import numpy as np
from numpy.typing import NDArray

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


def rms(a: NDArray, window_size: Optional[int] = None) -> NDArray:
    """
    Calculate RMS series for the given NumPy array.

    :param a: NumPy array to process.
    :param window_size: Window size for the RMS calculation. If not specified, it defaults to the length of the array.
    :return: A NumPy array containing the RMS series.
    """
    if 0 in a.shape:
        raise ValueError("Cannot input empty array")

    if window_size is None:
        window_size = a.shape[-1]

    if (
        a.dtype == np.dtype("float32")
        and (a.ndim == 1 or (a.ndim == 2 and a.shape[0] == 1))
        and (a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"])
    ):
        output_shape = a.shape[:-1] + (a.shape[-1] // window_size,)
        output_array = np.zeros(shape=output_shape, dtype=a.dtype)
        _numpy_rms.lib.rms(
            _numpy_rms.ffi.cast("float *", a.ctypes.data),
            window_size,
            _numpy_rms.ffi.cast("float *", output_array.ctypes.data),
            output_array.size,
        )
        return output_array

    from .fallback import rms_numpy

    return rms_numpy(a, window_size)
