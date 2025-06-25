// ────────────────────────────────────────────────────────────────────────────
//  main.cc ─ WAV → DSP → RKNN inference (FP32 input, no transpose)
//
//  • Sends the FP32 log-Mel tensor exactly as H=134, W=1, C=60 (NHWC).
//  • in.size is 134 × 60 × 4 = 32 160 B, matching nElems * sizeof(float).
//  • RKNN does the quantisation internally; no -5 error.
//
// ────────────────────────────────────────────────────────────────────────────
#include <rknn_api.h>
#include "dsp/melbank.h"
#include <sndfile.h>

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <cstring>
#include <algorithm>

#define BAR "============================================================\n"
#define SECTION(t) do { puts("\n" BAR t "\n" BAR); } while (0)
#define CHECK(x,msg) do { if(!(x)){ fprintf(stderr,"ERROR: %s\n",msg); exit(1);} } while (0)

static void enable_rknn_debug(int lvl = 3)
{
    char buf[2] = { char('0' + lvl), 0 };
    setenv("RKNN_LOG_LEVEL", buf, 1);
}

static bool load_wav(const char* p, std::vector<float>& out)
{
    SF_INFO i{}; SNDFILE* f = sf_open(p, SFM_READ, &i);
    if (!f) { perror("sf_open"); return false; }
    if (i.channels != 1 || i.samplerate != 16000) {
        fprintf(stderr,"Need mono 16-kHz WAV (got %d ch @ %d Hz)\n",
                i.channels,i.samplerate);
        sf_close(f); return false;
    }
    out.resize(static_cast<size_t>(i.frames));
    sf_readf_float(f, out.data(), i.frames);
    sf_close(f);
    return true;
}

static void dump_attr(const rknn_tensor_attr& a)
{
    printf("index=%u  fmt=%d  type=%d  size(B)=%u  scale=%g  zp=%d  "
           "dims=[%d,%d,%d,%d]\n",
           a.index,a.fmt,a.type,a.size,a.scale,a.zp,
           a.dims[0],a.dims[1],a.dims[2],a.dims[3]);
}

static void diag_param_invalid(const rknn_context& ctx,
                               const rknn_input&  in,
                               size_t host_sz)
{
    rknn_tensor_attr need{}; need.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &need, sizeof(need));

    puts("\n*** RKNN_ERR_PARAM_INVALID (-5) – details ********************");
    dump_attr(need);
    printf("We supplied : type=%d  fmt=%d  size=%u  pass_through=%d\n",
           in.type,in.fmt,in.size,in.pass_through);
    printf("Element check: host=%zu  model=%u  (%s)\n",
           host_sz,need.size,(host_sz==need.size?"OK":"MISMATCH!"));
    puts("**************************************************************\n");
}

static void print_versions(rknn_context ctx)
{
    rknn_sdk_version v{};
    if (!rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &v, sizeof(v)))
        printf("SDK  : %s   Driver : %s\n", v.api_version, v.drv_version);
}

int main(int argc,char** argv)
{
    if (argc<3){ fprintf(stderr,"usage: %s model.rknn audio.wav\n",argv[0]); return 1; }
    const char* model_path = argv[1];
    const char* wav_path   = argv[2];

    enable_rknn_debug(3);

    SECTION("1/6  WAV → RAM");
    std::vector<float> wav;
    CHECK(load_wav(wav_path,wav),"load_wav");
    printf("Loaded %zu samples\n", wav.size());

    SECTION("2/6  DSP (log-Mel)");
    auto mel = dsp::wav_to_logmel(wav);           // [time][mel_bins]
    const size_t T = mel.size();                  // 134
    const size_t M = mel[0].size();               // 60
    printf("Mel shape : %zu × %zu (time × mel)\n",T,M);

    SECTION("3/6  RKNN init");
    rknn_context ctx=0;
    CHECK(!rknn_init(&ctx,(void*)model_path,0,0,nullptr),"rknn_init");
    print_versions(ctx);

    rknn_tensor_attr in_attr{}; in_attr.index = 0;
    rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&in_attr,sizeof(in_attr));
    dump_attr(in_attr);                           // fmt = 1 (NHWC)

    SECTION("4/6  Prepare FP32 NHWC buffer");
    std::vector<float> buf_nhwc(T*M);             // H(134)×C(60)
    for(size_t h=0; h<T; ++h)                     // H = time dimension
        std::memcpy(&buf_nhwc[h*M], mel[h].data(), M*sizeof(float));

    rknn_input in{};
    in.index        = 0;
    in.buf          = buf_nhwc.data();
    in.size         = static_cast<uint32_t>(buf_nhwc.size()*sizeof(float)); // 32 160
    in.type         = RKNN_TENSOR_FLOAT32;        // runtime will quantise
    in.fmt          = RKNN_TENSOR_NHWC;
    in.pass_through = 0;

    int ret = rknn_inputs_set(ctx,1,&in);
    if (ret){
        fprintf(stderr,"rknn_inputs_set FAILED  ret=%d\n",ret);
        if (ret==-5) diag_param_invalid(ctx,in,in.size);
        rknn_destroy(ctx); return 1;
    }

    SECTION("5/6  Inference");
    CHECK(!rknn_run(ctx,nullptr),"rknn_run");
    puts("Inference OK");

    SECTION("6/6  Fetch output");
    rknn_output out{}; out.want_float = 1;
    CHECK(!rknn_outputs_get(ctx,1,&out,nullptr),"outputs_get");

    const float* e = static_cast<const float*>(out.buf);
    printf("Embedding[0..3] = %.3f  %.3f  %.3f  %.3f\n",
           e[0],e[1],e[2],e[3]);

    rknn_outputs_release(ctx,1,&out);
    rknn_destroy(ctx);
    return 0;
}
