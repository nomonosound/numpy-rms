import os
from cffi import FFI


ffibuilder = FFI()
ffibuilder.cdef("void rms_contiguous(float *, size_t);")

script_dir = os.path.dirname(os.path.realpath(__file__))
c_file_path = os.path.join(script_dir, "_rms.c")

with open(c_file_path, "r") as file:
    c_code = file.read()

extra_compile_args = ["-mavx", "-mavx512f", "-O3", "-Wall"]
if os.name == "posix":
    extra_compile_args.append("-Wextra")

ffibuilder.set_source("_numpy_rms", c_code, extra_compile_args=extra_compile_args)


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
