#pragma once
namespace Tmpl8 {
    __m128 crossSIMD(__m128 a, __m128 b) {
        __m128 a_yzx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 b_yzx = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c = _mm_sub_ps(
            _mm_mul_ps(a, b_yzx),
            _mm_mul_ps(a_yzx, b)
        );
        return _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 0, 2, 1));
    }
}
