#pragma once

#ifndef VINF_TARGET_RTX_3080_TI
#define VINF_TARGET_RTX_3080_TI 1
#endif

#define VINF_CUDA_ARCH_SM86 86
#define VINF_RTX_3080_TI_EXPECTED_SMS 80
#define VINF_RTX_3080_TI_VRAM_BYTES 12884901888ULL

// Consumer Ampere path. Hopper/Blackwell-only features must stay disabled.
#define VINF_AMPERE 1
#define VINF_DISABLE_TMA 1
#define VINF_DISABLE_HOPPER 1
#define VINF_DISABLE_BLACKWELL 1

namespace vinf {

struct rtx_3080_ti_config {
    static constexpr int cuda_arch = VINF_CUDA_ARCH_SM86;
    static constexpr int expected_sms = VINF_RTX_3080_TI_EXPECTED_SMS;
    static constexpr unsigned long long vram_bytes = VINF_RTX_3080_TI_VRAM_BYTES;

    static constexpr int instruction_width = 32;
    static constexpr int timing_width = 128;
    static constexpr int instruction_pipeline_stages = 2;
    static constexpr int consumer_warps = 8;
    static constexpr int role_warps = 4;
    static constexpr int num_warps = consumer_warps + role_warps;
    static constexpr int warp_threads = 32;
    static constexpr int num_threads = num_warps * warp_threads;
    static constexpr int page_size_bytes = 16 * 1024;
};

} // namespace vinf

