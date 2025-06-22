#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace dsp {

/* -------- float32 → IEEE-754 half (uint16) ------------------------------ */
inline uint16_t float_to_fp16(float f)
{
#if defined(__ARM_FP16_FORMAT_IEEE) && !defined(__clang__)
    return *reinterpret_cast<uint16_t*>(__builtin_convertfloat16(f));
#else
    /* portable scalar fallback (round-to-nearest-even) */
    union { float f; uint32_t u; } in{f};
    uint32_t sign =  (in.u >> 16) & 0x8000;
    uint32_t mant =  in.u & 0x007fffff;
    int32_t  exp  = ((in.u >> 23) & 0xff) - 127 + 15;
    if (exp <= 0) {                  /* subnormal / zero */
        if (exp < -10) return sign;  /* ±0 */
        mant |= 0x00800000;
        uint16_t val = mant >> (1 - exp + 13);
        return sign | val;
    }
    if (exp >= 31)                   /* inf / NaN */
        return sign | 0x7c00 | (mant ? 0x200 : 0);
    return sign | (exp << 10) | (mant >> 13);
#endif
}

/* -------- float32 → vector< uint16_t >  (row-major) --------------------- */
inline std::vector<uint16_t>
to_fp16(const std::vector<std::vector<float>>& m)
{
    std::vector<uint16_t> out;
    out.reserve(m.size() * m[0].size());
    for (const auto& row : m)
        for (float v : row)
            out.push_back(float_to_fp16(v));
    return out;          /* shape flattened to 1×1×60×134 later */
}

/* -------- float32 → int8/uint8 with (scale, zero_point) ------------------ */
template<typename INT>
std::vector<INT>
to_int8(const std::vector<std::vector<float>>& m,
        float scale, int32_t zero_point)
{
    static_assert(std::is_same<INT,int8_t>::value ||
                  std::is_same<INT,uint8_t>::value,
                  "INT must be int8_t or uint8_t");

    std::vector<INT> out;
    out.reserve(m.size() * m[0].size());

    const float inv = 1.f / scale;
    const int32_t qmin = std::is_signed<INT>::value ? -128 : 0;
    const int32_t qmax = std::is_signed<INT>::value ?  127 : 255;

    for (const auto& row : m)
        for (float v : row) {
            int32_t q = static_cast<int32_t>(std::round(v * inv)) + zero_point;
            q = std::min(std::max(q, qmin), qmax);
            out.push_back(static_cast<INT>(q));
        }
    return out;
}

} // namespace dsp
