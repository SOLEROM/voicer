#include "melbank.h"
#include "preemph.h"
#include "stft.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {

/* Helper – freq (Hz) → mel,  mel → freq */
inline float hz_to_mel(float f)  { return 2595.f * std::log10(1.f + f / 700.f); }
inline float mel_to_hz(float m)  { return 700.f * (std::pow(10.f, m / 2595.f) - 1.f); }

} // namespace

namespace dsp {

// ---------------------------------------------------------------- mel FB
std::vector<std::vector<float>>
build_mel_fb(size_t sr, size_t n_fft, size_t n_mels, float fmin, float fmax)
{
    const size_t n_bins = n_fft / 2 + 1;
    const float  mel_min = hz_to_mel(fmin);
    const float  mel_max = hz_to_mel(fmax);
    const float  mel_step = (mel_max - mel_min) / (n_mels + 1);

    /* centre freqs in mel scale */
    std::vector<float> mel_pts(n_mels + 2);
    for (size_t i = 0; i < mel_pts.size(); ++i)
        mel_pts[i] = mel_min + i * mel_step;

    /* convert to FFT bin numbers */
    std::vector<size_t> bin(n_mels + 2);
    for (size_t i = 0; i < bin.size(); ++i)
        bin[i] = static_cast<size_t>(std::floor((n_fft + 1) *
                     mel_to_hz(mel_pts[i]) / sr));

    /* filters */
    std::vector<std::vector<float>> fb(n_mels, std::vector<float>(n_bins, 0.f));
    for (size_t m = 1; m <= n_mels; ++m) {
        size_t b_left  = bin[m - 1];
        size_t b_center= bin[m];
        size_t b_right = bin[m + 1];

        for (size_t k = b_left; k < b_center; ++k)
            fb[m-1][k] = (k - b_left) / float(b_center - b_left);
        for (size_t k = b_center; k < b_right; ++k)
            fb[m-1][k] = (b_right - k) / float(b_right - b_center);
    }
    return fb;
}

// ---------------------------------------------------------------- wav → log-Mel
std::vector<std::vector<float>>
wav_to_logmel(const std::vector<float>& wav)
{
    constexpr size_t SR       = 16'000;
    constexpr size_t N_FFT    = 512;
    constexpr size_t WIN      = 400;     // 25 ms
    constexpr size_t HOP      = 240;     // 15 ms
    constexpr size_t N_MELS   = 60;
    constexpr size_t TARGET_T = 134;     // frames

    /* 1. pre-emphasis */
    std::vector<float> w = wav;
    preemph(w);

    /* 2. STFT power */
    auto spec = stft_power(w, SR, WIN, HOP, N_FFT);   // F × 257

    /* 3. mel filter bank (cached static) */
    static auto fb = build_mel_fb(SR, N_FFT, N_MELS);
    const size_t n_bins = N_FFT / 2 + 1;

    /* 4. spec → mel */
    std::vector<std::vector<float>> mel(spec.size(),
                                        std::vector<float>(N_MELS, 0.f));
    for (size_t t = 0; t < spec.size(); ++t)
        for (size_t m = 0; m < N_MELS; ++m)
            for (size_t k = 0; k < n_bins; ++k)
                mel[t][m] += spec[t][k] * fb[m][k];

    /* 5. log10 & eps */
    constexpr float EPS = 1e-6f;
    for (auto& frame : mel)
        for (float& v : frame)
            v = std::log10(std::max(v, EPS));

    /* 6. pad / crop to TARGET_T frames */
    if (mel.size() < TARGET_T)
        mel.resize(TARGET_T, std::vector<float>(N_MELS, 0.f));
    else if (mel.size() > TARGET_T)
        mel.erase(mel.begin() + TARGET_T, mel.end());

    /* 7. mean-normalise per feature */
    std::vector<float> mean(N_MELS, 0.f);
    for (size_t m = 0; m < N_MELS; ++m)
        for (size_t t = 0; t < TARGET_T; ++t)
            mean[m] += mel[t][m];
    for (float& v : mean) v /= TARGET_T;

    for (size_t m = 0; m < N_MELS; ++m)
        for (size_t t = 0; t < TARGET_T; ++t)
            mel[t][m] -= mean[m];

    return mel;   // shape 134 × 60  (time × mel)
}

} // namespace dsp
