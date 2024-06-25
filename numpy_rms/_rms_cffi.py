import os
import platform

from cffi import FFI

ffibuilder = FFI()
ffibuilder.cdef("void rms(float *, int, float *, size_t);")

script_dir = os.path.dirname(os.path.realpath(__file__))
c_file_path = os.path.join(script_dir, "_rms.c")

with open(c_file_path, "r") as file:
    c_code = file.read()

extra_compile_args = ["-O3", "-Wall"]
if os.name == "posix":
    extra_compile_args.append("-Wextra")

# Detect architecture and set appropriate SIMD-related compile args
if platform.machine().lower() in ["x86_64", "amd64", "i386", "i686"]:
    extra_compile_args.append("-mavx")
elif platform.machine().lower() in ["arm64", "aarch64"]:
    extra_compile_args.append("-march=armv8-a+simd")

ffibuilder.set_source("_numpy_rms", c_code, extra_compile_args=extra_compile_args)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
