#include <iostream>
#include <rknn_api.h>
#include <cstring>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " model.rknn" << std::endl;
        return -1;
    }

    const char* model_path = argv[1];

    // Load RKNN model from file
    FILE* fp = fopen(model_path, "rb");
    if (!fp) {
        std::cerr << "Failed to open " << model_path << std::endl;
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int model_size = ftell(fp);
    rewind(fp);
    void* model_data = malloc(model_size);
    fread(model_data, 1, model_size, fp);
    fclose(fp);

    // Init RKNN
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_init failed: " << ret << std::endl;
        return -1;
    }

    // Query SDK and driver version
    rknn_sdk_version version;
    if (rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(version)) == 0) {
        std::cout << "RKNN SDK version:   " << version.api_version << std::endl;
        std::cout << "Driver version:     " << version.drv_version << std::endl;
    }

    // Query input attributes
    int n_inputs = 0, n_outputs = 0;
    rknn_input_output_num io_num;
    if (rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) == 0) {
        n_inputs = io_num.n_input;
        n_outputs = io_num.n_output;
        std::cout << "Num inputs:  " << n_inputs << "\n";
        std::cout << "Num outputs: " << n_outputs << "\n";
    }

    for (int i = 0; i < n_inputs; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        std::cout << "\n[Input " << i << "] Name: " << attr.name
                  << ", Dims: [" << attr.dims[0] << "," << attr.dims[1] << "," << attr.dims[2] << "," << attr.dims[3] << "]"
                  << ", Type: " << attr.type << ", Qnt Type: " << attr.qnt_type
                  << ", Fmt: " << attr.fmt << "\n";
    }

    for (int i = 0; i < n_outputs; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        std::cout << "\n[Output " << i << "] Name: " << attr.name
                  << ", Dims: [" << attr.dims[0] << "," << attr.dims[1] << "," << attr.dims[2] << "," << attr.dims[3] << "]"
                  << ", Type: " << attr.type << ", Qnt Type: " << attr.qnt_type
                  << ", Fmt: " << attr.fmt << "\n";
    }

    rknn_destroy(ctx);
    free(model_data);
    return 0;
}
