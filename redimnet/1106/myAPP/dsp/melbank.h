#pragma once
#include <vector>
#include <cstddef>

namespace dsp {

/* Build a linear‐mel filter bank (60 bands) – one-time cost at start-up. */
std::vector<std::vector<float>>
build_mel_fb(size_t sr = 16'000, size_t n_fft = 512,
             size_t n_mels = 60, float fmin = 0.f, float fmax = 8'000.f);

/* End-to-end: WAV → log-Mel (60 × 134) float32 matrix.
   Uses DSP modules: preemph → STFT → mel FB → log10 → pad/crop → norm. */
std::vector<std::vector<float>>
wav_to_logmel(const std::vector<float>& wav);

} // namespace dsp
