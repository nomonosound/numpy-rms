import numpy as np
import pytest

import numpy_rms


class TestRMS:
    def test_rms_even(self):
        arr = np.arange(40, dtype=np.float32)
        rms = numpy_rms.rms(arr, window_size=10)
        assert rms.shape == (4,)
        print(rms)

        # TODO: more assert...
