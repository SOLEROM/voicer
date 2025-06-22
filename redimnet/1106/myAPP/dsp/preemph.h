#pragma once
#include <vector>
#include <cstdint>

namespace dsp {

template<typename T>
inline void preemph(std::vector<T>& wav, T alpha = static_cast<T>(0.97))
{
    if (wav.empty()) return;
    for (size_t i = wav.size() - 1; i > 0; --i)
        wav[i] = wav[i] - alpha * wav[i - 1];
    wav[0] *= (1 - alpha);          // match Python behaviour
}

} // namespace dsp
