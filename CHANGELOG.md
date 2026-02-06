# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-12-27

### Added

* Add support for Python 3.14

### Removed

* Remove support for Python 3.9

## [0.6.0] - 2025-06-29

### Added

* Add support for Python 3.13

## [0.5.0] - 2025-03-14

### Changed

* Bump numpy to >=2,<3. If you need compatibility with numpy 1.*, you can use numpy-rms==0.4.2

### Removed

* Remove support for Python 3.8
* Remove support for PyPy on Windows

## [0.4.2] - 2024-07-13

### Changed

* Optimize the processing of multichannel arrays

## [0.4.1] - 2024-07-09

### Fixed

* Fix multichannel processing in fallback function

## [0.4.0] - 2024-07-07

### Added

* Add macOS builds

## [0.3.0] - 2024-06-25

### Added

* Add Linux builds compiled for ARM with NEON SIMD optimizations

## [0.2.0] - 2024-05-29

### Changes

* Make `window_size` optional. When not specified, it defaults to the length of the given array.

## [0.1.0] - 2024-04-11

Initial release
