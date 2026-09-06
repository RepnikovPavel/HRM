#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "gemm_family.cuh"

__global__ void rms_scale_kernel(const bf16* __restrict__ h, const float* __restrict__ sumsq,
                                 bf16* __restrict__ y, float* __restrict__ rstd,
                                 long total, int D, float eps) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float r = rsqrtf(sumsq[i / D] / D + eps);
    y[i] = __float2bfloat16(__bfloat162float(h[i]) * r);
    if (i % D == 0) rstd[i / D] = r;
}

torch::Tensor swiglu_forward(torch::Tensor x, torch::Tensor w_gate, torch::Tensor w_up, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(w_gate.is_contiguous() && w_up.is_contiguous());
    TORCH_CHECK(w_gate.dtype() == torch::kBFloat16 && w_up.dtype() == torch::kBFloat16);
    int M = x.size(0), K = x.size(1), N = w_gate.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8 for 16B async copies");
    auto y = torch::empty({M, N}, x.options());
    tn_launch<1, 2>((int)cfg, (const bf16*)x.data_ptr(), (const bf16*)w_gate.data_ptr(),
                    (const bf16*)w_up.data_ptr(), nullptr, (bf16*)y.data_ptr(), nullptr, nullptr,
                    M, N, K, at::cuda::getCurrentCUDAStream());
    return y;
}

torch::Tensor linear_fwd(torch::Tensor x, torch::Tensor w, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(w.is_contiguous() && w.dtype() == torch::kBFloat16);
    int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8 for 16B async copies");
    auto y = torch::empty({M, N}, x.options());
    tn_launch<0, 1>((int)cfg, (const bf16*)x.data_ptr(), (const bf16*)w.data_ptr(), nullptr,
                    nullptr, (bf16*)y.data_ptr(), nullptr, nullptr, M, N, K,
                    at::cuda::getCurrentCUDAStream());
    return y;
}

std::vector<torch::Tensor> linear_resid_rmsnorm_fwd(torch::Tensor x, torch::Tensor w,
                                                    torch::Tensor resid, double eps, int64_t cfg) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(w.is_contiguous() && w.dtype() == torch::kBFloat16);
    TORCH_CHECK(resid.is_contiguous() && resid.dtype() == torch::kBFloat16);
    int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8 for 16B async copies");
    TORCH_CHECK(resid.size(0) == M && resid.size(1) == N);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto h = torch::empty({M, N}, x.options());
    auto sumsq = torch::zeros({M}, x.options().dtype(torch::kFloat));
    tn_launch<2, 1>((int)cfg, (const bf16*)x.data_ptr(), (const bf16*)w.data_ptr(), nullptr,
                    (const bf16*)resid.data_ptr(), (bf16*)h.data_ptr(), nullptr,
                    (float*)sumsq.data_ptr(), M, N, K, stream);
    auto y = torch::empty({M, N}, x.options());
    auto rstd = torch::empty({M}, x.options().dtype(torch::kFloat));
    long total = (long)M * N;
    int blocks = (int)((total + 255) / 256);
    rms_scale_kernel<<<blocks, 256, 0, stream>>>(
        (const bf16*)h.data_ptr(), (const float*)sumsq.data_ptr(), (bf16*)y.data_ptr(),
        (float*)rstd.data_ptr(), total, N, (float)eps);
    return {y, rstd};
}
