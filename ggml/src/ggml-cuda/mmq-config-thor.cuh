// DRIVE Thor-U (sm_100/101/103/110) Dedicated MMQ Configuration Table
// Optimizations:
// 1. Occupancy = 2 for NVFP4 to maximize 228KB SMEM multi-block latency hiding.
// 2. Fallback to Ampere config for other standard quantized formats.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_thor(ggml_type type, int J, bool fallback) {
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, true);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, true);

    CASE(GGML_TYPE_NVFP4, 256, 2, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, true, false);

    return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
}
