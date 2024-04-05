from setuptools import setup

setup(
    cffi_modules=["numpy_rms/_rms_cffi.py:ffibuilder"],
)
