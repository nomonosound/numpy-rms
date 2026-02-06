# numpy-rms: a fast function for calculating a series of Root Mean Square (RMS) values

* Written in C and takes advantage of AVX (on x86-64) or NEON (on ARM) for speed
* The fast implementation is tailored for C-contiguous 1-dimensional and 2-dimensional float32 arrays

# Installation

[![PyPI version](https://img.shields.io/pypi/v/numpy-rms.svg?style=flat)](https://pypi.org/project/numpy-rms/)
![python 3.10, 3.11, 3.12. 3.13, 3.14](https://img.shields.io/badge/Python-3.10%20|%203.11|%203.12%20|%203.13%20|%203.14-blue)
![os: Linux, macOS, Windows](https://img.shields.io/badge/OS-Linux%20%28arm%20%26%20x86--64%29%20|%20macOS%20%28arm%20%26%20x86--64%29%20|%20Windows%20%28x86--64%29-blue)

```
$ pip install numpy-rms
```

# Usage

```py
import numpy_rms
import numpy as np

arr = np.arange(40, dtype=np.float32)
rms_series = numpy_rms.rms(arr, window_size=10)
print(rms_series.shape)  # (4,)
```

# Changelog

## [0.7.0] - 2025-12-27

### Added

* Add support for Python 3.14

### Removed

* Remove support for Python 3.9

For the complete changelog, go to [CHANGELOG.md](CHANGELOG.md)

# Development

* Install dev/build/test dependencies as denoted in pyproject.toml
* `CC=clang pip install -e .`
* `pytest`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
