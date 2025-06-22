#pragma once
#include <vector>
#include <cstddef>

namespace dsp {

/* Build a 1-D Hamming window of length win_len (float32). */
std::vector<float> hamming(size_t win_len);

/* Compute power-spectrum frames (|FFT|²) from mono PCM.
   - wav:      input mono float [-1..1]
   - sr:       sample-rate (Hz) – must equal _SR (16 kHz) for now
   - win_len:  window length  (default 400 samples = 25 ms @16 k)
   - hop:      hop size        (default 240 samples = 15 ms)
   - n_fft:    FFT size        (default 512)  */
std::vector<std::vector<float>>
stft_power(const std::vector<float>& wav,
           size_t sr       = 16'000,
           size_t win_len  = 400,
           size_t hop      = 240,
           size_t n_fft    = 512);

} // namespace dsp
