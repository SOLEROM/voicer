#include "stft.h"
#include <cmath>
#include <algorithm>
#include <cassert>
#include "kiss_fftr.h" 

namespace dsp {

// ─────────────────────── Hamming window
std::vector<float> hamming(size_t N)
{
    std::vector<float> w(N);
    const float twopi = 6.283185307f;
    for (size_t n = 0; n < N; ++n)
        w[n] = 0.54f - 0.46f * std::cos(twopi * n / (N - 1));
    return w;
}

// ─────────────────────── STFT → power spectrum
std::vector<std::vector<float>>
stft_power(const std::vector<float>& wav,
           size_t sr, size_t win_len, size_t hop, size_t n_fft)
{
    assert(sr == 16'000 && "only 16-kHz audio supported");

    const auto win = hamming(win_len);

    const size_t n_frames =
        wav.size() < win_len ? 1 :
        1 + (wav.size() - win_len) / hop;

    std::vector<std::vector<float>> out(
        n_frames, std::vector<float>(n_fft/2 + 1));

    /* KissFFT real-FFT plan */
    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_fft, /*inverse=*/0, nullptr, nullptr);

    std::vector<float>         buf_time(n_fft, 0.f);
    std::vector<kiss_fft_cpx>  buf_freq(n_fft/2 + 1);

    for (size_t i = 0; i < n_frames; ++i)
    {
        size_t offset = i * hop;

        /* window & zero-pad */
        std::fill(buf_time.begin(), buf_time.end(), 0.f);
        for (size_t n = 0; n < win_len; ++n)
            buf_time[n] = wav[offset + n] * win[n];

        kiss_fftr(cfg, buf_time.data(), buf_freq.data());

        /* magnitude² (bins 0 … n_fft/2) */
        for (size_t k = 0; k <= n_fft/2; ++k) {
            float re = buf_freq[k].r;
            float im = buf_freq[k].i;
            out[i][k] = re*re + im*im;
        }
    }
    kiss_fftr_free(cfg);
    return out;
}

} // namespace dsp
