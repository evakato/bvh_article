#pragma once

namespace Tmpl8 {
    inline void TransformPositionAVX(__m256& x, __m256& y, __m256& z, const mat4& M) {
        // Load matrix elements into AVX registers
        __m256 m0 = _mm256_set1_ps(M.cell[0]);
        __m256 m1 = _mm256_set1_ps(M.cell[1]);
        __m256 m2 = _mm256_set1_ps(M.cell[2]);
        __m256 m3 = _mm256_set1_ps(M.cell[3]);
        __m256 m4 = _mm256_set1_ps(M.cell[4]);
        __m256 m5 = _mm256_set1_ps(M.cell[5]);
        __m256 m6 = _mm256_set1_ps(M.cell[6]);
        __m256 m7 = _mm256_set1_ps(M.cell[7]);
        __m256 m8 = _mm256_set1_ps(M.cell[8]);
        __m256 m9 = _mm256_set1_ps(M.cell[9]);
        __m256 m10 = _mm256_set1_ps(M.cell[10]);
        __m256 m11 = _mm256_set1_ps(M.cell[11]);
        __m256 m12 = _mm256_set1_ps(M.cell[12]);
        __m256 m13 = _mm256_set1_ps(M.cell[13]);
        __m256 m14 = _mm256_set1_ps(M.cell[14]);
        __m256 m15 = _mm256_set1_ps(M.cell[15]);

        // Perform matrix multiplication
        __m256 new_x = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(m0, x),
                    _mm256_mul_ps(m1, y)),
                _mm256_mul_ps(m2, z)),
            m3);

        __m256 new_y = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(m4, x),
                    _mm256_mul_ps(m5, y)),
                _mm256_mul_ps(m6, z)),
            m7);

        __m256 new_z = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(m8, x),
                    _mm256_mul_ps(m9, y)),
                _mm256_mul_ps(m10, z)),
            m11);

        // Update the original x, y, z with the new values
        x = new_x;
        y = new_y;
        z = new_z;
    }

    inline void TransformVectorAVX(__m256& x, __m256& y, __m256& z, const mat4& M) {
        // Load matrix elements into AVX registers
        __m256 m0 = _mm256_set1_ps(M.cell[0]);
        __m256 m1 = _mm256_set1_ps(M.cell[1]);
        __m256 m2 = _mm256_set1_ps(M.cell[2]);
        __m256 m4 = _mm256_set1_ps(M.cell[4]);
        __m256 m5 = _mm256_set1_ps(M.cell[5]);
        __m256 m6 = _mm256_set1_ps(M.cell[6]);
        __m256 m8 = _mm256_set1_ps(M.cell[8]);
        __m256 m9 = _mm256_set1_ps(M.cell[9]);
        __m256 m10 = _mm256_set1_ps(M.cell[10]);

        // Perform matrix multiplication
        __m256 new_x = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m0, x),
                _mm256_mul_ps(m1, y)),
            _mm256_mul_ps(m2, z));

        __m256 new_y = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m4, x),
                _mm256_mul_ps(m5, y)),
            _mm256_mul_ps(m6, z));

        __m256 new_z = _mm256_add_ps(
            _mm256_add_ps(
                _mm256_mul_ps(m8, x),
                _mm256_mul_ps(m9, y)),
            _mm256_mul_ps(m10, z));

        // Update the original x, y, z with the new values
        x = new_x;
        y = new_y;
        z = new_z;
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

    inline __m256 modAVX(__m256 a, __m256 b) {
        __m256 div = _mm256_div_ps(a, b);
        __m256 floored_div = _mm256_floor_ps(div);
        __m256 mult = _mm256_mul_ps(floored_div, b);
        __m256 result = _mm256_sub_ps(a, mult);
        return result;
    }

    static uint64_t seed = 123456789;

    // Linear congruential generator parameters
    #define LCG_A 6364136223846793005ULL
    #define LCG_C 1ULL

    inline __m256i generate_random_ints() {
        __m256i state = _mm256_set_epi64x(seed, seed + 1, seed + 2, seed + 3);
        __m256i multiplier = _mm256_set1_epi64x(LCG_A);
        __m256i increment = _mm256_set1_epi64x(LCG_C);

        seed = LCG_A * seed + LCG_C;

        state = _mm256_mul_epu32(state, multiplier);
        state = _mm256_add_epi64(state, increment);

        return state;
    }

    inline __m256 generate_random_floats() {
        __m256i random_ints = generate_random_ints();
        __m128i lo = _mm256_castsi256_si128(random_ints); // lower 128 bits
        __m128i hi = _mm256_extracti128_si256(random_ints, 1); // upper 128 bits
        __m256i random_ints_32 = _mm256_set_m128i(hi, lo);
        __m256 float_divisor = _mm256_set1_ps(1.0f / 4294967296.0f);
        __m256 random_floats = _mm256_mul_ps(_mm256_cvtepi32_ps(random_ints_32), float_divisor);
        return random_floats;
    }

    inline __m256 avxLength(__m256 x, __m256 y, __m256 z) {
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 y2 = _mm256_mul_ps(y, y);
        __m256 z2 = _mm256_mul_ps(z, z);
        __m256 sum = _mm256_add_ps(_mm256_add_ps(x2, y2), z2);
        __m256 length = _mm256_sqrt_ps(sum);

        return length;
    }

}
