#pragma once

namespace Tmpl8 {
    inline __m128 crossSIMD(__m128 a, __m128 b) {
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(
            _mm_mul_ps(a, b_yzx),
            _mm_mul_ps(a_yzx, b)
        );
        return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
    }

    inline __m128 normalizeSIMD(__m128 v) {
        __m128 dotProduct = _mm_dp_ps(v, v, 0xFF);
        __m128 invSqrt = _mm_rsqrt_ps(dotProduct);
        __m128 normalized = _mm_mul_ps(v, invSqrt);
        return normalized;
    }

    inline void TransformPositionAVX(__m256* x, __m256* y, __m256* z, const mat4& transform) {
        __m256 m0 = _mm256_set_ps(transform(0, 3), transform(0, 2), transform(0, 1), transform(0, 0),
            transform(0, 3), transform(0, 2), transform(0, 1), transform(0, 0));
        __m256 m1 = _mm256_set_ps(transform(1, 3), transform(1, 2), transform(1, 1), transform(1, 0),
            transform(1, 3), transform(1, 2), transform(1, 1), transform(1, 0));
        __m256 m2 = _mm256_set_ps(transform(2, 3), transform(2, 2), transform(2, 1), transform(2, 0),
            transform(2, 3), transform(2, 2), transform(2, 1), transform(2, 0));
        __m256 m3 = _mm256_set_ps(transform(3, 3), transform(3, 2), transform(3, 1), transform(3, 0),
            transform(3, 3), transform(3, 2), transform(3, 1), transform(3, 0));

        __m256 temp_x = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2)), m3);
        __m256 temp_y = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2)), m3);
        __m256 temp_z = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2)), m3);

        *x = temp_x;
        *y = temp_y;
        *z = temp_z;
    }

    inline void TransformVectorAVX(__m256* x, __m256* y, __m256* z, const mat4& transform) {
        __m256 m0 = _mm256_set_ps(transform(0, 3), transform(0, 2), transform(0, 1), transform(0, 0),
            transform(0, 3), transform(0, 2), transform(0, 1), transform(0, 0));
        __m256 m1 = _mm256_set_ps(transform(1, 3), transform(1, 2), transform(1, 1), transform(1, 0),
            transform(1, 3), transform(1, 2), transform(1, 1), transform(1, 0));
        __m256 m2 = _mm256_set_ps(transform(2, 3), transform(2, 2), transform(2, 1), transform(2, 0),
            transform(2, 3), transform(2, 2), transform(2, 1), transform(2, 0));

        __m256 temp_x = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2));
        __m256 temp_y = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2));
        __m256 temp_z = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(*x, m0), _mm256_mul_ps(*y, m1)), _mm256_mul_ps(*z, m2));

        *x = temp_x;
        *y = temp_y;
        *z = temp_z;
    }

    inline void crossAVX(__m256 a_x, __m256 a_y, __m256 a_z, __m256 b_x, __m256 b_y, __m256 b_z, __m256* out_x, __m256* out_y, __m256* out_z) {
        *out_x = _mm256_sub_ps(_mm256_mul_ps(a_y, b_z), _mm256_mul_ps(a_z, b_y));
        *out_y = _mm256_sub_ps(_mm256_mul_ps(a_z, b_x), _mm256_mul_ps(a_x, b_z));
        *out_z = _mm256_sub_ps(_mm256_mul_ps(a_x, b_y), _mm256_mul_ps(a_y, b_x));
    }

    inline __m256 dotAVX(__m256 a_x, __m256 a_y, __m256 a_z, __m256 b_x, __m256 b_y, __m256 b_z) {
        __m256 mul_x = _mm256_mul_ps(a_x, b_x);
        __m256 mul_y = _mm256_mul_ps(a_y, b_y);
        __m256 mul_z = _mm256_mul_ps(a_z, b_z);
        __m256 sum_xy = _mm256_add_ps(mul_x, mul_y);
        __m256 dot = _mm256_add_ps(sum_xy, mul_z);
        return dot;
    }

    inline __m256 absAVX(__m256 input) {
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
        return _mm256_and_ps(input, mask);
    }

    inline void normalizeAVX(__m256* x, __m256* y, __m256* z) {
        __m256 x2 = _mm256_mul_ps(*x, *x);
        __m256 y2 = _mm256_mul_ps(*y, *y);
        __m256 z2 = _mm256_mul_ps(*z, *z);
        __m256 length2 = _mm256_add_ps(_mm256_add_ps(x2, y2), z2);
        __m256 invLength = _mm256_rsqrt_ps(length2);
        *x = _mm256_mul_ps(*x, invLength);
        *y = _mm256_mul_ps(*y, invLength);
        *z = _mm256_mul_ps(*z, invLength);
    }

    inline void print_m256(__m256 var) {
        float values[8];
        _mm256_storeu_ps(values, var);
        printf("Values: %f %f %f %f %f %f %f %f\n",
            values[0], values[1], values[2], values[3],
            values[4], values[5], values[6], values[7]);
    }

    inline __m256 atan_avx_approximation(__m256 x) {
        __m256 a1 = _mm256_set1_ps(0.99997726f);
        __m256 a3 = _mm256_set1_ps(-0.33262347f);
        __m256 a5 = _mm256_set1_ps(0.19354346f);
        __m256 a7 = _mm256_set1_ps(-0.11643287f);
        __m256 a9 = _mm256_set1_ps(0.05265332f);
        __m256 a11 = _mm256_set1_ps(-0.01172120f);

        __m256 x_sq = _mm256_mul_ps(x, x);
        __m256 result;
        result = a11;
        result = _mm256_fmadd_ps(x_sq, result, a9);
        result = _mm256_fmadd_ps(x_sq, result, a7);
        result = _mm256_fmadd_ps(x_sq, result, a5);
        result = _mm256_fmadd_ps(x_sq, result, a3);
        result = _mm256_fmadd_ps(x_sq, result, a1);
        result = _mm256_mul_ps(x, result);

        return result;
    }

    inline __m256 atan2_avx(__m256 y, __m256 x) {
        __m256 zero = _mm256_setzero_ps();
        __m256 pi = _mm256_set1_ps(3.14159265358979323846f);
        __m256 pi_half = _mm256_set1_ps(3.14159265358979323846f / 2);
        __m256 neg_pi_half = _mm256_set1_ps(-3.14159265358979323846f / 2);

        // atan(y / x)
        __m256 ratio = _mm256_div_ps(y, x);
        __m256 angle = atan_avx_approximation(ratio);

        // Mask for x < 0
        __m256 mask_x_neg = _mm256_cmp_ps(x, zero, _CMP_LT_OS);
        // Mask for y < 0
        __m256 mask_y_neg = _mm256_cmp_ps(y, zero, _CMP_LT_OS);

        // Adjust angle for x < 0
        __m256 angle_adjusted = _mm256_blendv_ps(angle, _mm256_add_ps(angle, pi), mask_x_neg);
        angle_adjusted = _mm256_blendv_ps(angle_adjusted, _mm256_sub_ps(angle, pi), mask_x_neg);

        // Final adjustment for y < 0 when x == 0
        __m256 mask_x_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OS);
        __m256 y_adjusted = _mm256_blendv_ps(pi_half, neg_pi_half, mask_y_neg);
        angle_adjusted = _mm256_blendv_ps(angle_adjusted, y_adjusted, mask_x_zero);

        return angle_adjusted;
    }

    inline __m256 modAVX(__m256 a, __m256 b) {
        __m256 div = _mm256_div_ps(a, b);
        __m256 floored_div = _mm256_floor_ps(div);
        __m256 mult = _mm256_mul_ps(floored_div, b);
        __m256 result = _mm256_sub_ps(a, mult);
        return result;
    }

}
