import numpy as np
import pytest

import numpy_rms


class TestRMS:
    def test_rms_even(self):
        arr = np.array([0.0, 1.0, -2.0, 0.0], dtype=np.float32)
        rms = numpy_rms.rms(arr)
        # TODO: assert...
