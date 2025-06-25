// test_dummy.cc  — minimal RKNN smoke‑test for Toolkit‑2.3.x headers
// ---------------------------------------------------------------------------
// 1) Reads tensor‑0 attributes (type / fmt / dims / size).
// 2) Allocates a random buffer with *matching byte‑size*.
// 3) Sends it with the correct pass_through value:    
//        • INT8 / UINT8  → pass_through = 1 (already quantised)  
//        • otherwise     → pass_through = 0 (runtime quantises)  
// 4) On RKNN_ERR_PARAM_INVALID (‑5) prints a verbose diff.
//
// NOTE: Toolkit‑2.3.x `rknn_input` has only {index, buf, size, pass_through,
//       type, fmt}.  Newer fields like `n_dims`, `dims[]`, `scale` … don’t
//       exist in these headers, so we don’t touch them.
// ---------------------------------------------------------------------------
#include <rknn_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

// ───────────────────────── helpers ──────────────────────────
#define CHECK(call, msg)                                              \
    do {                                                             \
        int _ret = (call);                                           \
        if (_ret != 0) {                                             \
            std::fprintf(stderr, "ERROR: %s (ret=%d)\n", msg, _ret); \
            std::exit(EXIT_FAILURE);                                 \
        }                                                            \
    } while (0)

static inline size_t bytes_per_type(int t)
{
    switch (t) {
        case RKNN_TENSOR_FLOAT16: return 2;
        case RKNN_TENSOR_FLOAT32: return 4;
        case RKNN_TENSOR_INT8:
        case RKNN_TENSOR_UINT8:  return 1;
        case RKNN_TENSOR_INT16:
        case RKNN_TENSOR_UINT16: return 2;
        case RKNN_TENSOR_INT32:
        case RKNN_TENSOR_UINT32: return 4;
        default: return 1;
    }
}

static void dump_attr(const char* tag, const rknn_tensor_attr& a)
{
    std::printf("%s: idx=%u fmt=%d type=%d size=%u scale=%g zp=%d dims=[%d %d %d %d]\n",
                tag, a.index, a.fmt, a.type, a.size, a.scale, a.zp,
                a.dims[0], a.dims[1], a.dims[2], a.dims[3]);
}

static void diag_param_invalid(rknn_context ctx,
                               const rknn_input& host,
                               size_t host_bytes)
{
    rknn_tensor_attr need{}; need.index = 0;
    rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &need, sizeof(need));

    std::puts("\n*** RKNN_ERR_PARAM_INVALID (-5) – details ********************");
    dump_attr("Model expects", need);
    std::printf("Host supplies : type=%d fmt=%d size=%u pass_through=%d\n",
                host.type, host.fmt, host.size, host.pass_through);
    std::printf("Element check : host=%zu  model=%u  (%s)\n",
                host_bytes, need.size, (host_bytes == need.size ? "OK" : "MISMATCH!"));
    std::puts("**************************************************************\n");
}

// ───────────────────────── main ─────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s model.rknn\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1. Initialise runtime
    rknn_context ctx = 0;
    CHECK(rknn_init(&ctx, (void*)argv[1], 0, 0, nullptr), "rknn_init");

    // 2. Query input‑0 attributes
    rknn_tensor_attr in_attr{}; in_attr.index = 0;
    CHECK(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
          "query input attr");
    dump_attr("Input", in_attr);

    const bool is_quant = (in_attr.type == RKNN_TENSOR_INT8 ||
                           in_attr.type == RKNN_TENSOR_UINT8);
    const int  pass_through = is_quant ? 1 : 0;

    size_t bytes_elem = bytes_per_type(in_attr.type);
    size_t elems      = 1;
    for (int i = 0; i < 4 && in_attr.dims[i] > 0; ++i)
        elems *= in_attr.dims[i];
    size_t buf_bytes = elems * bytes_elem;

    // 3. Allocate + randomise buffer
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(buf_bytes));
    if (!buf) { std::perror("malloc"); return EXIT_FAILURE; }

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    if (bytes_elem == 1) {
        for (size_t i = 0; i < buf_bytes; ++i)
            buf[i] = static_cast<uint8_t>(std::rand() & 0xFF);
    } else {
        float* f = reinterpret_cast<float*>(buf);
        for (size_t i = 0; i < elems; ++i)
            f[i] = static_cast<float>(std::rand()) / RAND_MAX * 2.f - 1.f;
    }

    // 4. Populate rknn_input
    rknn_input rin{};
    rin.index        = 0;
    rin.buf          = buf;
    rin.size         = static_cast<uint32_t>(buf_bytes);
    rin.pass_through = pass_through;
    rin.type         = static_cast<rknn_tensor_type>(in_attr.type);
    rin.fmt          = in_attr.fmt;

    int ret = rknn_inputs_set(ctx, 1, &rin);
    if (ret != 0) {
        std::fprintf(stderr, "ERROR: rknn_inputs_set failed (ret=%d)\n", ret);
        if (ret == -5) diag_param_invalid(ctx, rin, buf_bytes);
        std::free(buf);
        rknn_destroy(ctx);
        return EXIT_FAILURE;
    }

    // 5. Run inference
    CHECK(rknn_run(ctx, nullptr), "rknn_run");

    // 6. Fetch output‑0 attr & data
    rknn_tensor_attr out_attr{}; out_attr.index = 0;
    CHECK(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)),
          "query output attr");
    dump_attr("Output", out_attr);

    rknn_output rout{}; rout.want_float = 1;
    CHECK(rknn_outputs_get(ctx, 1, &rout, nullptr), "outputs_get");

    std::printf("\nOutput[0] byte‑size : %u\n", rout.size);
    if (rout.size >= 4 * sizeof(float)) {
        const float* f = static_cast<const float*>(rout.buf);
        std::printf("First 4 values      : %.3f  %.3f  %.3f  %.3f\n", f[0], f[1], f[2], f[3]);
    }

    // 7. Cleanup
    rknn_outputs_release(ctx, 1, &rout);
    std::free(buf);
    rknn_destroy(ctx);

    std::puts("\nInference completed OK – dummy smoke test passed.");
    return EXIT_SUCCESS;
}
