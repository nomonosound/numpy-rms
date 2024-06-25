#include <float.h>
#include <math.h>
#include <stdbool.h>

void rms_avx2(const float *a, int window_size, float *rms_output, size_t output_length);
void rms_neon(const float *a, int window_size, float *rms_output, size_t output_length);
void rms_scalar(const float *a, int window_size, float *rms_output, size_t output_length);

void rms(const float *a, int window_size, float *rms_output, size_t output_length) {
    #if defined(__x86_64__) || defined(_M_X64)
        rms_avx2(a, window_size, rms_output, output_length);
    #elif defined(__arm__) || defined(__aarch64__)
        rms_neon(a, window_size, rms_output, output_length);
    #else
        rms_scalar(a, window_size, rms_output, output_length);
    #endif
}

// AVX2 implementation
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

// NB! The length of a must be >= output_length * window_size
void rms_avx2(const float *a, int window_size, float *rms_output, size_t output_length) {
    int i = 0;
    int j;
    double window_sum;
    __m256 sum_vec;
    __m256 vals;
    __m256 vals_squared;
    float temp[8];
    int window_end;

    const int remainder_after_avx2 = window_size % 8;

    for (int output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;

        sum_vec = _mm256_setzero_ps();

        window_end = i + window_size;

        // First, process elements that will not fit into a full AVX2 register
        if (remainder_after_avx2 != 0) {
            int remainder_end = i + remainder_after_avx2;
            for (; i < remainder_end; i++) {
                window_sum += a[i] * a[i];
            }
        }

        for (; i < window_end; i += 8) {
            vals = _mm256_loadu_ps(a + i);
            vals_squared = _mm256_mul_ps(vals, vals);
            sum_vec = _mm256_add_ps(sum_vec, vals_squared);
        }

        _mm256_storeu_ps(temp, sum_vec);
        for (j = 0; j < 8; ++j) {
            window_sum += temp[j];
        }

        // Compute RMS for the window and assign it to the output
        rms_output[output_i] = sqrt(window_sum / window_size);
    }
}

// NEON implementation
#elif defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>

// NB! The length of a must be >= output_length * window_size
void rms_neon(const float *a, int window_size, float *rms_output, size_t output_length) {
    int i = 0;
    int j;
    double window_sum;
    float32x4_t sum_vec;
    float32x4_t vals;
    float32x4_t vals_squared;
    float temp[4];
    int window_end;

    const int remainder_after_neon = window_size % 4;

    for (int output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;

        sum_vec = vdupq_n_f32(0.0f);

        window_end = i + window_size;

        // First, process elements that will not fit into a full NEON register
        if (remainder_after_neon != 0) {
            int remainder_end = i + remainder_after_neon;
            for (; i < remainder_end; i++) {
                window_sum += a[i] * a[i];
            }
        }

        for (; i < window_end; i += 4) {
            vals = vld1q_f32(a + i);
            vals_squared = vmulq_f32(vals, vals);
            sum_vec = vaddq_f32(sum_vec, vals_squared);
        }

        vst1q_f32(temp, sum_vec);
        for (j = 0; j < 4; ++j) {
            window_sum += temp[j];
        }

        // Compute RMS for the window and assign it to the output
        rms_output[output_i] = (float)sqrt(window_sum / window_size);
    }
}

#else

// Scalar implementation (fallback for other architectures)
void rms_scalar(const float *a, int window_size, float *rms_output, size_t output_length) {
    double window_sum;
    for (size_t output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;
        for (int i = output_i * window_size; i < (output_i + 1) * window_size; i++) {
            window_sum += a[i] * a[i];
        }
        rms_output[output_i] = (float)sqrt(window_sum / window_size);
    }
}
#endif
