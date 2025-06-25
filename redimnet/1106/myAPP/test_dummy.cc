// dummyInfer.c  — minimal RKNN model smoke-test (C/C++)
//
// Usage:
//     ./dummyInfer model.rknn
//
// Loads the RKNN model, queries the input tensor, fills it with random
// values, runs inference, and prints the output tensor dimensions and
// size in bytes.  Good for “garbage-in / garbage-out” sanity checks.
//-------------------------------------------------------------------------
// Build examples:
//   C (recommended):
//       arm-linux-gnueabihf-gcc dummyInfer.c -I${RKNN_INC} -L${RKNN_LIB} \
//            -lrknnmrt -lpthread -lm -ldl -o dummyInfer
//
//   C++ (if you change the suffix to .cc/.cpp):
//       arm-linux-gnueabihf-g++ dummyInfer.cc -I${RKNN_INC} -L${RKNN_LIB} \
//            -lrknnmrt -lpthread -lm -ldl -o dummyInfer
//-------------------------------------------------------------------------
#include <rknn_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>

#define CHECK(call,msg)                                         \
    do {                                                        \
        int _ret = (call);                                      \
        if (_ret != 0) {                                        \
            std::fprintf(stderr, "ERROR: %s (ret=%d)\n", msg, _ret); \
            std::exit(EXIT_FAILURE);                            \
        }                                                       \
    } while (0)

static inline size_t bytes_per_type(int type)
{
    switch (type) {
        case RKNN_TENSOR_FLOAT16: return 2;
        case RKNN_TENSOR_FLOAT32: return 4;
        case RKNN_TENSOR_INT8:
        case RKNN_TENSOR_UINT8:  return 1;
        case RKNN_TENSOR_INT16:
        case RKNN_TENSOR_UINT16: return 2;
        case RKNN_TENSOR_INT32:
        case RKNN_TENSOR_UINT32: return 4;
        default: return 1;               // fallback
    }
}

static void print_attr(const char* tag, const rknn_tensor_attr& a)
{
    std::printf("%s: idx=%u fmt=%d type=%d size=%u dims=[%d %d %d %d]\n",
                tag, a.index, a.fmt, a.type, a.size,
                a.dims[0], a.dims[1], a.dims[2], a.dims[3]);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s model.rknn\n", argv[0]);
        return EXIT_FAILURE;
    }

    // 1) Init
    rknn_context ctx = 0;
    CHECK(rknn_init(&ctx, (void*)argv[1], 0, 0, nullptr), "rknn_init");

    // 2) Inspect input tensor 0
    rknn_tensor_attr in_attr{}; in_attr.index = 0;
    CHECK(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)),
          "query input attr");
    print_attr("Input", in_attr);

    size_t in_bytes = in_attr.size;
    if (in_bytes == 0) {                     // some toolchains leave size=0
        in_bytes = bytes_per_type(in_attr.type);
        for (int i = 0; i < 4 && in_attr.dims[i] > 0; ++i)
            in_bytes *= in_attr.dims[i];
    }

    // 3) Allocate & fill random buffer (cast needed for C++)
    auto* rand_buf = static_cast<uint8_t*>(std::malloc(in_bytes));
    if (!rand_buf) { std::perror("malloc"); return EXIT_FAILURE; }

    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (size_t i = 0; i < in_bytes; ++i)
        rand_buf[i] = static_cast<uint8_t>(std::rand() & 0xFF);

    rknn_input in{};
    in.index        = 0;
    in.buf          = rand_buf;
    in.size         = static_cast<uint32_t>(in_bytes);
    in.pass_through = 0;
    in.type         = in_attr.type;
    in.fmt          = in_attr.fmt;

    CHECK(rknn_inputs_set(ctx, 1, &in), "inputs_set");

    // 4) Inference
    CHECK(rknn_run(ctx, nullptr), "rknn_run");

    // 5) Output tensor 0 attr + first few floats
    rknn_tensor_attr out_attr{}; out_attr.index = 0;
    CHECK(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)),
          "query output attr");
    print_attr("Output", out_attr);

    rknn_output out{}; out.want_float = 1;
    CHECK(rknn_outputs_get(ctx, 1, &out, nullptr), "outputs_get");

    std::printf("\nOutput[0] byte-size : %u\n", out.size);
    if (out.size >= 4 * sizeof(float)) {
        const float* f = static_cast<const float*>(out.buf);
        std::printf("First 4 values      : %.3f  %.3f  %.3f  %.3f\n",
                    f[0], f[1], f[2], f[3]);
    }

    rknn_outputs_release(ctx, 1, &out);
    std::free(rand_buf);
    rknn_destroy(ctx);

    puts("\nInference completed OK – dummy smoke test passed.");
    return EXIT_SUCCESS;
}