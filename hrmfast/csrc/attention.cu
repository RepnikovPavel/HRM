#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_kernels.cuh"

std::vector<torch::Tensor> attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale) {
    TORCH_CHECK(q.is_cuda() && q.dtype() == torch::kBFloat16);
    TORCH_CHECK(q.size(3) == ATTN_D && q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1);
    TORCH_CHECK(q.stride(2) == ATTN_D && k.stride(2) == ATTN_D && v.stride(2) == ATTN_D);
    int B = q.size(0), S = q.size(1), H = q.size(2);
    TORCH_CHECK(S <= 128);
    int sp = sp_of(S);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto o = torch::empty({B, S, H, ATTN_D}, q.options());
    auto lse = torch::empty({B * H, S}, q.options().dtype(torch::kFloat));
    const int smem = 3 * sp * ATTN_D * 2;

#define FWD_CASE(SPV) \
    { \
        static bool cfgd = false; \
        if (!cfgd) { \
            cudaFuncSetAttribute(attn_fwd_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            cfgd = true; \
        } \
        attn_fwd_kernel<SPV><<<B * H, 128, smem, stream>>>( \
            (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(), \
            (bf16*)o.data_ptr(), (float*)lse.data_ptr(), S, H, (float)scale, \
            q.stride(0), k.stride(0), v.stride(0), \
            q.stride(1), k.stride(1), v.stride(1)); \
    }
    ATTN_DISPATCH(sp, FWD_CASE);
#undef FWD_CASE
    return {o, lse};
}

std::vector<torch::Tensor> attn_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                         torch::Tensor o, torch::Tensor dout, torch::Tensor lse,
                                         double scale) {
    int B = q.size(0), S = q.size(1), H = q.size(2);
    int sp = sp_of(S);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto dq = torch::empty({B, S, H, ATTN_D}, q.options());
    auto dk = torch::empty({B, S, H, ATTN_D}, q.options());
    auto dv = torch::empty({B, S, H, ATTN_D}, q.options());
    auto dsum = torch::empty({B * H, S}, q.options().dtype(torch::kFloat));
    const int smem1 = 4 * sp * ATTN_D * 2;
    const int smem2 = (2 * 32 * ATTN_D + 6 * 16 * ATTN_D + 2 * 16 * 40) * 2;
    const int nst = sp / 32;

#define BWD_CASE(SPV) \
    { \
        static bool cfgd1 = false, cfgd2 = false; \
        if (!cfgd1) { \
            cudaFuncSetAttribute(attn_bwd_dq_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem1); \
            cfgd1 = true; \
        } \
        if (!cfgd2) { \
            cudaFuncSetAttribute(attn_bwd_dkdv_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem2); \
            cfgd2 = true; \
        } \
        attn_bwd_dq_kernel<SPV><<<B * H, 128, smem1, stream>>>( \
            (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(), \
            (const bf16*)o.data_ptr(), (const bf16*)dout.data_ptr(), (const float*)lse.data_ptr(), \
            (bf16*)dq.data_ptr(), (float*)dsum.data_ptr(), S, H, (float)scale, \
            q.stride(0), k.stride(0), v.stride(0), o.stride(0), dout.stride(0), \
            q.stride(1), k.stride(1), v.stride(1), o.stride(1), dout.stride(1)); \
        dim3 g2(nst, B * H); \
        attn_bwd_dkdv_kernel<SPV><<<g2, 128, smem2, stream>>>( \
            (const bf16*)q.data_ptr(), (const bf16*)k.data_ptr(), (const bf16*)v.data_ptr(), \
            (const bf16*)dout.data_ptr(), (const float*)lse.data_ptr(), (const float*)dsum.data_ptr(), \
            (bf16*)dk.data_ptr(), (bf16*)dv.data_ptr(), S, H, (float)scale, \
            q.stride(0), k.stride(0), v.stride(0), dout.stride(0), \
            q.stride(1), k.stride(1), v.stride(1), dout.stride(1)); \
    }
    ATTN_DISPATCH(sp, BWD_CASE);
#undef BWD_CASE
    return {dq, dk, dv};
}
