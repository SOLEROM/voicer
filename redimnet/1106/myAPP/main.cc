// main.cc  –  full pipeline: WAV → DSP → quant → RKNN
#include "rknn_api.h"
#include "dsp/melbank.h"
#include "dsp/quant.h"
#include <sndfile.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>

// ---------- WAV loader (16-kHz mono) ---------------------------------------
static bool load_wav(const char* path, std::vector<float>& out)
{
    SF_INFO inf{};  SNDFILE* sf = sf_open(path, SFM_READ, &inf);
    if (!sf) { perror("sf_open"); return false; }
    if (inf.channels!=1 || inf.samplerate!=16000) {
        fprintf(stderr,"Need mono 16-k WAV\n"); sf_close(sf); return false;
    }
    out.resize(inf.frames);
    sf_readf_float(sf,out.data(),inf.frames);
    sf_close(sf);
    return true;
}

// ---------------------------------------------------------------------------
int main(int argc,char**argv)
{
    if (argc<3){printf("usage: %s model.rknn audio.wav\n",argv[0]);return 1;}
    const char* model_path=argv[1]; const char* wav_path=argv[2];

    // 1) WAV ---------------------------------------------------------------
    std::vector<float> wav; if(!load_wav(wav_path,wav)) return 1;
    printf("Loaded %zu samples\n",wav.size());

    // 2) DSP  --------------------------------------------------------------
    auto mel = dsp::wav_to_logmel(wav);           // 134×60 float32

    // 3) RKNN init ---------------------------------------------------------
    rknn_context ctx; if(rknn_init(&ctx,(void*)model_path,0,0,nullptr)){
        fprintf(stderr,"rknn_init failed\n"); return 1;}

    rknn_tensor_attr in_attr{}; in_attr.index=0;
    rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&in_attr,sizeof(in_attr));

    // 4) Quant -------------------------------------------------------------
    std::vector<uint16_t> buf16;
    std::vector<int8_t>   buf8 ;

    rknn_input in{0};
    in.index = 0;
    in.fmt   = RKNN_TENSOR_NCHW;
    in.pass_through = 0;            // let runtime de-quant internally

#if defined(RKNN_TENSOR_FLOAT16)
    if (in_attr.type==RKNN_TENSOR_FLOAT16){
        buf16=dsp::to_fp16(mel);
        in.type = RKNN_TENSOR_FLOAT16;
        in.buf  = buf16.data();
        in.size = buf16.size()*sizeof(uint16_t);
    } else
#endif
    {   /* INT8 / UINT8 branch                         */
        float   scale = in_attr.scale ? in_attr.scale : 0.02f;
        int32_t zp    = 0;     
        buf8 = dsp::to_int8<int8_t>(mel,scale,zp);
        in.type = in_attr.type;             // INT8 or UINT8
        in.buf  = buf8.data();
        in.size = buf8.size();
    }
    rknn_inputs_set(ctx,1,&in);

    // 5) run ---------------------------------------------------------------
    rknn_run(ctx,nullptr);

    rknn_output out{0}; out.want_float=1;
    rknn_outputs_get(ctx,1,&out,nullptr);
    float* e = static_cast<float*>(out.buf);
    printf("Embedding[0..3] = %.3f  %.3f  %.3f  %.3f\n",e[0],e[1],e[2],e[3]);
    rknn_outputs_release(ctx,1,&out);
    rknn_destroy(ctx);
    return 0;
}
