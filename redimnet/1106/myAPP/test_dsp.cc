// test_dsp.cc  –  unit-tests for DSP stage 2A/2B
#include "dsp/preemph.h"
#include "dsp/stft.h"

#include <cstdio>
#include <vector>
#include <cmath>

//------------------------------------------------------------------
//  2A – pre-emphasis
//------------------------------------------------------------------
static bool test_preemph()
{
    std::vector<float> wav = {1.0f, 2.0f, 3.0f};
    dsp::preemph(wav, 0.97f);

    const float expected = 0.03f;
    const float eps      = 1e-4f;
    bool ok = std::fabs(wav[0] - expected) < eps;

    std::printf("[preemph] w[0]=%.5f (%.5f) → %s\n",
                wav[0], expected, ok ? "PASS" : "FAIL");
    return ok;
}

//------------------------------------------------------------------
//  2B – STFT power-spectrum
//------------------------------------------------------------------
static bool test_stft()
{
    constexpr float pi = 3.14159265358979323846f;
    const float sr    = 16'000.f;
    const float freq  = 1'000.f;

    /* generate 320-sample sine */
    std::vector<float> wav(320);
    for (size_t n = 0; n < wav.size(); ++n)
        wav[n] = std::sin(2 * pi * freq * n / sr);

    auto spec = dsp::stft_power(wav);          // default params
    if (spec.empty())
        return false;

    float sum0 = 0.f;
    for (float p : spec[0]) sum0 += p;
    bool ok = sum0 > 1e-3f;

    std::printf("[stft] Σpower(frame0) = %.3f → %s\n",
                sum0, ok ? "PASS" : "FAIL");
    return ok;
}

//------------------------------------------------------------------
int main()
{
    bool ok1 = test_preemph();
    bool ok2 = test_stft();

    std::printf("═══════════════════════════════════════════\n");
    std::printf("DSP tests %s\n", (ok1 && ok2) ? "PASSED" : "FAILED");
    return (ok1 && ok2) ? 0 : 1;
}
