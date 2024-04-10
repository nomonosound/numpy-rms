#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdbool.h>

void rms_scalar(const float *a, size_t length, int window_size, float *rms_output, size_t output_length) {
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

void rms(const float *a, size_t length, int window_size, float *rms_output, size_t output_length) {
    int i = 0;
    double window_sum;
    __m256 sum_vec;
    __m256 vals;
    __m256 vals_squared;
    float temp[8];

    for (int output_i = 0; output_i < output_length; output_i++) {
        window_sum = 0.0;

        sum_vec = _mm256_setzero_ps();

        int window_end = i + window_size;
        for (; i < window_end; i += 8) {
            vals = _mm256_loadu_ps(a + i);
            vals_squared = _mm256_mul_ps(vals, vals);
            sum_vec = _mm256_add_ps(sum_vec, vals_squared);
        }

        _mm256_storeu_ps(temp, sum_vec);
        for (int j = 0; j < 8; ++j) {
            window_sum += temp[j];
        }

        rms_output[output_i] = sqrt(window_sum / window_size);
    }
}
