#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>

void rms(const float *a, size_t length, int window_size, float *rms_output, size_t output_length) {
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
