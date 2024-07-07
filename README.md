# numpy-rms: a fast function for calculating a series of Root Mean Square (RMS) values

* Written in C and takes advantage of AVX (on x86-64) or NEON (on ARM) for speed
* The fast implementation is tailored for contiguous 1-dimensional float32 arrays

# Installation

[![PyPI version](https://img.shields.io/pypi/v/numpy-rms.svg?style=flat)](https://pypi.org/project/numpy-rms/)
![python 3.8, 3.9, 3.10, 3.11, 3.12](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10%20|%203.11|%203.12-blue)
![os: Linux, macOS, Windows](https://img.shields.io/badge/OS-Linux%20|%20macOS%20|%20Windows-blue)
![CPU: x86_64 & arm64](https://img.shields.io/badge/CPU-x86__64%20|%20arm64-blue)

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

## [0.4.0] - 2024-07.07

### Added

* Add macOS builds

For the complete changelog, go to [CHANGELOG.md](CHANGELOG.md)

# Development

* Install dev/build/test dependencies as denoted in pyproject.toml
* `CC=clang pip install -e .`
* `pytest`

# Acknowledgements

This library is maintained/backed by [Nomono](https://nomono.co/), a Norwegian audio AI startup.
