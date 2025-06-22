// main.cc  –  RKNN probe (no DSP tests)
#include "rknn_api.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char* argv[])
{
    const char* model_path = (argc > 1) ? argv[1] : nullptr;

    printf("═══════════════════════════════════════════\n");
    if (model_path)
        printf("Loading model: %s\n", model_path);
    else
        printf("No model supplied – calling rknn_init(NULL)…\n");

    rknn_context ctx;
    int ret = rknn_init(&ctx,
                        const_cast<char*>(model_path),  // C API takes void*
                        0,                              // size == 0 → file path
                        0,
                        nullptr);

    printf("rknn_init returned %d\n", ret);

    if (ret == 0)
        rknn_destroy(ctx);

    return (ret == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
