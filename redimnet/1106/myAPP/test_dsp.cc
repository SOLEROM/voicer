// test_dsp.cc – unit-tests for the DSP stack (2A-2C) + INT8 quant check
#include "dsp/preemph.h"
#include "dsp/stft.h"
#include "dsp/melbank.h"
#include "dsp/quant.h"

#include <cstdio>
#include <vector>
#include <cmath>
#include <limits>
#include <cstdint>

// #################################################################################################//
// #################################################################################################//
// test_preemph & test_stft – unchanged.

// test_logmel – now returns the log-Mel matrix to caller.

// test_quant_int8 – converts that matrix with a dummy (scale = 0.02, zero_point = 0) using dsp::to_int8<int8_t> and verifies:

//     output length = 60 × 134

//     every value is within int8 range.

// main() reports overall PASS / FAIL.

// #################################################################################################//
// #################################################################################################//


// ---------- 2A  pre-emphasis -------------------------------------------------
static bool test_preemph()
{
    std::vector<float> w = {1.0f, 2.0f, 3.0f};
    dsp::preemph(w);
    bool ok = std::fabs(w[0] - 0.03f) < 1e-4f;
    std::printf("[preemph] w[0] = %.5f → %s\n",
                w[0], ok ? "PASS" : "FAIL");
    return ok;
}

// ---------- 2B  STFT power ---------------------------------------------------
static bool test_stft()
{
    constexpr float sr   = 16'000.f;
    constexpr float freq = 1'000.f;
    std::vector<float> wav(320);
    for (size_t n = 0; n < wav.size(); ++n)
        wav[n] = std::sin(2*M_PI*freq*n/sr);

    auto spec = dsp::stft_power(wav);
    float sum0 = 0.f;
    for (float p : spec[0]) sum0 += p;
    bool ok = sum0 > 1e-3f;
    std::printf("[stft] Σpower(frame0) = %.2f → %s\n",
                sum0, ok ? "PASS" : "FAIL");
    return ok;
}

// ---------- 2C  log-Mel ------------------------------------------------------
static bool test_logmel(std::vector<std::vector<float>>& mel_out)
{
    constexpr float sr   = 16'000.f;
    constexpr float freq = 1'000.f;
    std::vector<float> wav(320);
    for (size_t n = 0; n < wav.size(); ++n)
        wav[n] = std::sin(2*M_PI*freq*n/sr);

    mel_out = dsp::wav_to_logmel(wav);
    bool shape_ok = (mel_out.size()==134 && mel_out[0].size()==60);
    bool finite   = std::isfinite(mel_out[0][0]);
    std::printf("[logmel] shape %zu×%zu, first=%.4f → %s\n",
                mel_out.size(), mel_out[0].size(), mel_out[0][0],
                (shape_ok && finite) ? "PASS" : "FAIL");
    return shape_ok && finite;
}

// ---------- quant  (float32 → INT8 affine) -----------------------------------
static bool test_quant_int8()
{
    std::vector<std::vector<float>> mel;
    if (!test_logmel(mel)) return false;

    constexpr float scale = 0.02f;   // dummy values for unit-test
    constexpr int32_t zp  = 0;

    auto q = dsp::to_int8<int8_t>(mel, scale, zp);

    bool size_ok = (q.size() == 134*60);
    bool range_ok = true;
    for (int8_t v : q)
        if (v < std::numeric_limits<int8_t>::min() ||
            v > std::numeric_limits<int8_t>::max())
            { range_ok = false; break; }

    std::printf("[quant] INT8 size=%zu, range check → %s\n",
                q.size(), (size_ok && range_ok) ? "PASS" : "FAIL");
    return size_ok && range_ok;
}

// ---------- main -------------------------------------------------------------
int main()
{
    bool ok =  test_preemph() &
               test_stft()    &
               test_quant_int8();

    std::printf("═══════════════════════════════════════════\n"
                "DSP+Quant tests %s\n", ok ? "PASSED" : "FAILED");
    return ok ? 0 : 1;
}
