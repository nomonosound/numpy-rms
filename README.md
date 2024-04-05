# numpy-rms: a fast function for calculating a series of RMS values

NumPy lacked an optimized rms function, so we wrote our own. At Nomono, we use it for audio processing, but it can be applied any kind of float32 ndarray.

* Written in C and takes advantage of AVX/AVX512 for speed
* Roughly **2.3x speedup** compared to the numpy amin+amax equivalent (tested on Intel CPU with numpy 1.24-1.26)
* The fast implementation is tailored for float32 arrays that are C-contiguous, F-contiguous or 1D strided. Strided arrays with ndim >= 2 get processed with numpy.amin and numpy.amax, so no perf gain there.

# Installation

[![PyPI version](https://img.shields.io/pypi/v/numpy-rms.svg?style=flat)](https://pypi.org/project/numpy-rms/)
![python 3.8, 3.9, 3.10, 3.11, 3.12](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11|%203.12-blue)
![os: Linux, Windows](https://img.shields.io/badge/OS-Linux%20|%20Windows-blue)
![CPU: x86_84](https://img.shields.io/badge/CPU-x86__64-blue)

```
$ pip install numpy-rms
```

# Usage

```py
import numpy_rms
import numpy as np

arr = np.arange(1337, dtype=np.float32)
rms_series = numpy_rms.rms(arr)
```

# Changelog

See [CHANGELOG.md](CHANGELOG.md)

# Development

* Install dev/build/test dependencies as denoted in pyproject.toml
* `CC=clang pip install -e .`
* `pytest`

# Running benchmarks

* `python scripts/perf_benchmark.py`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
