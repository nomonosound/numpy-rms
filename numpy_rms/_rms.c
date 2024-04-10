#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>

#ifdef _MSC_VER
    #include <intrin.h>  // MSVC
#else
    #include <cpuid.h>  // GCC and Clang
#endif

void rms(const float *a, size_t length, int window_size, float *rms_output, size_t output_length) {
    int i = 0;
    int window_end;
    double window_sum;

    for (int output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;
        window_end = i + window_size;
        for (; i < window_end; i++) {
            window_sum += a[i] * a[i];
        }
        rms_output[output_i] = sqrt(window_sum / window_size);
    }
    return;
}
