#include <float.h>
#include <immintrin.h>
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


void rms(const float *a, size_t length) {
    // TODO
    return;
}

void rms_avx(const float *a, size_t length) {
    // TODO
    return;
}

void rms_contiguous(const float *a, size_t length) {
    // Return early for empty arrays
    if (length == 0) {
        return;
    }
    // TODO IF length and AVX, use avx (or avx512)
    rms(a, length);
    return;
}
