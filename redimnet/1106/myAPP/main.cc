// main.cc  –  WAV → DSP → quant → RKNN   (verbose edition)
#include <rknn_api.h>
#include "dsp/melbank.h"
#include "dsp/quant.h"
#include <sndfile.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstring>

// ──────────── pretty console helpers ────────────
#define BAR   "============================================================\n"
#define SECTION(txt) do { puts("\n" BAR txt "\n" BAR); } while (0)
#define CHECK(x, msg) do { if(!(x)){fprintf(stderr,"ERROR: %s\n",msg); exit(1);} } while(0)

// ──────────── turn on RKNN internal logs ─────────
static void enable_rknn_debug(int level = 3)        // 0-5 (5 = trace)
{
    char buf[2] = { char('0' + level), '\0' };
    setenv("RKNN_LOG_LEVEL", buf, 1);

    /* ► extra dump switches – enable if your Toolkit build supports them
       setenv("RKNN_DUMP_TENSOR", "1", 1);
       setenv("RKNN_DUMP_LAYER",  "1", 1);
    */
}

// ──────────── WAV loader (16-kHz mono) ───────────
static bool load_wav(const char* path, std::vector<float>& out)
{
    SF_INFO info{};
    SNDFILE* sf = sf_open(path, SFM_READ, &info);
    if (!sf) { perror("sf_open"); return false; }

    if (info.channels != 1 || info.samplerate != 16000) {
        fprintf(stderr, "Need mono 16-k WAV (got %d ch @ %d Hz)\n",
                info.channels, info.samplerate);
        sf_close(sf);
        return false;
    }
    out.resize(static_cast<size_t>(info.frames));
    sf_readf_float(sf, out.data(), info.frames);
    sf_close(sf);
    return true;
}

// ──────────── print tensor attributes nicely ─────
static void dump_attr(const rknn_tensor_attr& a)
{
    printf("index=%u  name=%s  fmt=%d  type=%d  qnt=%d  dims=[%d,%d,%d,%d]\n"
           "size=%u  scale=%g  zp=%d\n",
           a.index, a.name, a.fmt, a.type, a.qnt_type,
           a.dims[0], a.dims[1], a.dims[2], a.dims[3],
           a.size,   a.scale,   a.zp);
}

// ──────────── main ───────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s  model.rknn  audio.wav\n", argv[0]);
        return 1;
    }
    const char* model_path = argv[1];
    const char* wav_path   = argv[2];

    enable_rknn_debug(5);               // same as Python’s “LOG_LEVEL = 3”

    SECTION("1/6  WAV → RAM");
    std::vector<float> wav;
    CHECK(load_wav(wav_path, wav), "loading WAV failed");
    printf("Loaded %zu samples from %s\n", wav.size(), wav_path);

    SECTION("2/6  DSP  (log-Mel)");
    auto mel = dsp::wav_to_logmel(wav);                    // [T, 60] f32
    printf("Mel shape : %zu × %zu  (time × n_mels)\n",
           mel.size(), mel.empty() ? 0 : mel[0].size());

    SECTION("3/6  RKNN  init");
    rknn_context ctx = 0;
    CHECK(rknn_init(&ctx, (void*)model_path, 0, 0, nullptr) == 0,
          "rknn_init failed");
    puts("Model loaded.");

    // query some meta info
    rknn_sdk_version ver{};
    rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    printf("SDK %.*s  Driver %.*s\n",
           int(sizeof(ver.api_version)), ver.api_version,
           int(sizeof(ver.drv_version)), ver.drv_version);

    uint32_t io_num[2];
    rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, io_num, sizeof(io_num));
    printf("IO : %u input(s), %u output(s)\n", io_num[0], io_num[1]);

    rknn_tensor_attr in_attr{};
    in_attr.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr));
    dump_attr(in_attr);

    SECTION("4/6  Quantise input");
    float   scale = in_attr.scale ? in_attr.scale : 0.02f;
    int32_t zp    = in_attr.zp;
    std::vector<int8_t> buf8 = dsp::to_int8<int8_t>(mel, scale, zp);

    printf("Host buf8.size = %zu  in_attr.size = %u  (%s)\n",
        buf8.size(), in_attr.size,
        buf8.size() == in_attr.size ? "OK" : "MISMATCH!");

    rknn_input in{};
    in.index        = 0;
    in.type         = in_attr.type;      // 2  →  INT8
    in.fmt          = in_attr.fmt;       // 1  →  NHWC
    in.size         = static_cast<uint32_t>(buf8.size());   // 8040 bytes
    in.buf          = buf8.data();
    in.pass_through = 1;                 // ★ we pass INT8 exactly as-is

    printf("Sending input : index=%u  type=%d  fmt=%d  size=%u  buf=%p\n",
        in.index, in.type, in.fmt, in.size, in.buf);

    int ret = rknn_inputs_set(ctx, 1, &in);
    if (ret != 0) {
        fprintf(stderr, "rknn_inputs_set FAILED  ret=%d  → ", ret);
        switch (ret) {                          // numeric map from rknn_api.h
            case -1:  fputs("generic failure\n",          stderr); break;
            case -2:  fputs("timeout\n",                  stderr); break;
            case -3:  fputs("device/context unavailable\n", stderr); break;
            case -4:  fputs("host memory allocation failed\n", stderr); break;
            case -5:  fputs("invalid parameter (dtype / fmt / size)\n", stderr); break;
            case -6:  fputs("invalid or corrupted model\n", stderr); break;
            case -7:  fputs("invalid context handle\n",    stderr); break;
            case -8:  fputs("input mismatch (shape / layout / type)\n", stderr); break;
            case -9:  fputs("output mismatch (rare here)\n", stderr); break;
            default:  fputs("unrecognized RKNN error\n",   stderr); break;
        }

        if (ret == -5 ) {
            fprintf(stderr, "rknn_inputs_set FAILED with  ret=%d  (-5 = bad parameter)\n", ret);

            // 1️⃣  show what the model really wants
            rknn_tensor_attr need{};
            need.index = 0;
            rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &need, sizeof(need));
            printf("\nModel expects :\n"
                "  type=%d  fmt=%d  size=%u  qnt_type=%d  scale=%g  zp=%d\n"
                "  dims (n=%d) : [%d %d %d %d]\n",
                need.type, need.fmt, need.size, need.qnt_type,
                need.scale, need.zp,
                need.n_dims,
                need.dims[0], need.dims[1], need.dims[2], need.dims[3]);

            // 2️⃣  show what we sent
            printf("\nWe supplied  :\n"
                "  type=%d  fmt=%d  size=%u  pass_through=%d  buf=%p\n",
                in.type,  in.fmt,  in.size, in.pass_through, in.buf);

            // 3️⃣  compare element count
            size_t expect_elems = need.size;               // bytes (quant) or floats
            printf("\nElement check: host=%zu  model=%zu  (%s)\n",
                static_cast<size_t>(in.size), expect_elems,
                in.size == expect_elems ? "OK" : "MISMATCH!");

            // 4️⃣  check data range (helps spot INT8 vs UINT8 misuse)
            if (in.buf && in.size) {
                const int8_t* p = static_cast<const int8_t*>(in.buf);
                auto mm = std::minmax_element(p, p + in.size);
                printf("Input range  : [%d … %d]\n\n", int(*mm.first), int(*mm.second));
            }
        }

        return 1;
    }
    printf("rknn_inputs_set OK");

    SECTION("5/6  Inference");
    CHECK(rknn_run(ctx, nullptr) == 0, "rknn_run failed");
    printf("Inference OK.");

    SECTION("6/6  Fetch output");
    rknn_output out{};
    out.want_float = 1;                       // de-quant to FP32
    CHECK(rknn_outputs_get(ctx, 1, &out, nullptr) == 0,
          "outputs_get failed");

    const float* e = static_cast<const float*>(out.buf);
    printf("Embedding[0..3] = %.3f %.3f %.3f %.3f\n",
           e[0], e[1], e[2], e[3]);

    // tidy up
    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
    return 0;
}
