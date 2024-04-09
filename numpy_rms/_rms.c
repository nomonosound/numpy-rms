#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>

#ifdef _MSC_VER
    #include <intrin.h>  // MSVC
#else
    #include <cpuid.h>  // GCC and Clang
#endif

#ifndef bit_AVX512F
#define bit_AVX512F     (1 << 16)
#endif

bool system_supports_avx512() {
    unsigned int eax, ebx, ecx, edx;

    // EAX=7, ECX=0: Extended Features
    #ifdef _MSC_VER
        // MSVC
        int cpuInfo[4];
        __cpuid(cpuInfo, 7);
        ebx = cpuInfo[1];
    #else
        // GCC, Clang
        __cpuid(7, eax, ebx, ecx, edx);
    #endif

    // Check the AVX512F bit in EBX
    return (ebx & bit_AVX512F) != 0;
}


void rms(const float *a, size_t length, int window_size, float *rms_output, size_t output_length) {
    int output_i;
    int i = 0;
    int j;
    double window_sum;
    double mean_squared;

    for (output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;
        for (j = i; j < i + window_size; j++) {
            window_sum += a[j] * a[j];
        }
        rms_output[output_i] = sqrt(window_sum / window_size);
        i += window_size;
    }
    return;
}
