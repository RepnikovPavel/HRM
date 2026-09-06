#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "gemm_family.cuh"

// backward of y = h * rstd with h = y / rstd substituted (h is never saved):
// dh_j = r*dy_j - (r/D) * y_j * dot(dy, y), r = rstd[row]; warp per row.
__global__ void rmsnorm_bwd_kernel(const bf16* __restrict__ dy, const bf16* __restrict__ y,
                                   const float* __restrict__ rstd, bf16* __restrict__ dh,
                                   int M, int D) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= M) return;
    int lane = threadIdx.x & 31;
    const bf16* dyr = dy + (long)row * D;
    const bf16* yr = y + (long)row * D;
    float dot = 0.f;
    for (int c = lane * 8; c < D; c += 32 * 8) {
        uint4 dv = *(const uint4*)(dyr + c);
        uint4 yv = *(const uint4*)(yr + c);
        const bf16* dvp = (const bf16*)&dv;
        const bf16* yvp = (const bf16*)&yv;
#pragma unroll
        for (int i = 0; i < 8; ++i) dot += __bfloat162float(dvp[i]) * __bfloat162float(yvp[i]);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
    float r = rstd[row];
    float coef = r / D * dot;
    bf16* dhr = dh + (long)row * D;
    for (int c = lane * 8; c < D; c += 32 * 8) {
        uint4 dv = *(const uint4*)(dyr + c);
        uint4 yv = *(const uint4*)(yr + c);
        const bf16* dvp = (const bf16*)&dv;
        const bf16* yvp = (const bf16*)&yv;
        uint4 ov;
        bf16* ovp = (bf16*)&ov;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            ovp[i] = __float2bfloat16(r * __bfloat162float(dvp[i]) - coef * __bfloat162float(yvp[i]));
        *(uint4*)(dhr + c) = ov;
    }
}

__global__ void dw_finalize_kernel(const float* __restrict__ ws, bf16* __restrict__ dw, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dw[i] = __float2bfloat16(ws[i]);
}

std::vector<torch::Tensor> swiglu_backward(torch::Tensor x, torch::Tensor w_gate, torch::Tensor w_up,
                                           torch::Tensor dout, int64_t cfg_tn, int64_t cfg_nn, int64_t cfg_nt) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16 && x.is_contiguous());
    TORCH_CHECK(dout.is_contiguous() && dout.dtype() == torch::kBFloat16);
    TORCH_CHECK(w_gate.is_contiguous() && w_up.is_contiguous());
    int M = x.size(0), K = x.size(1), N = w_gate.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    // dg/du are chunked over M: at full batch they would be the largest transient
    // allocation in the step (0.8 GB at gbs 3072/GPU2), and VRAM is the constraint.
    const long MC = 65536;
    auto dg = torch::empty({std::min((long)M, MC), N}, x.options());
    auto du = torch::empty({std::min((long)M, MC), N}, x.options());
    auto dx = torch::empty({M, K}, x.options());
    auto dwg = torch::empty({N, K}, x.options());
    auto dwu = torch::empty({N, K}, x.options());
    auto ws0 = torch::zeros({N, K}, x.options().dtype(torch::kFloat));
    auto ws1 = torch::zeros({N, K}, x.options().dtype(torch::kFloat));

    for (long m0 = 0; m0 < M; m0 += MC) {
        int mc = (int)std::min((long)M - m0, MC);
        const bf16* xc = (const bf16*)x.data_ptr() + m0 * K;
        const bf16* dc = (const bf16*)dout.data_ptr() + m0 * N;
        tn_launch<3, 2>((int)cfg_tn, xc, (const bf16*)w_gate.data_ptr(),
                        (const bf16*)w_up.data_ptr(), dc,
                        (bf16*)dg.data_ptr(), (bf16*)du.data_ptr(), nullptr, mc, N, K, stream);
        nn_launch<2>((int)cfg_nn, (const bf16*)dg.data_ptr(), (const bf16*)du.data_ptr(),
                     (const bf16*)w_gate.data_ptr(), (const bf16*)w_up.data_ptr(),
                     (bf16*)dx.data_ptr() + m0 * K, mc, N, K, stream);
        nt_launch<2>((int)cfg_nt, (const bf16*)dg.data_ptr(), (const bf16*)du.data_ptr(), xc,
                     (float*)ws0.data_ptr(), (float*)ws1.data_ptr(), mc, N, K, stream);
    }

    long nkw = (long)N * K;
    int blocks = (int)((nkw + 255) / 256);
    dw_finalize_kernel<<<blocks, 256, 0, stream>>>((const float*)ws0.data_ptr(), (bf16*)dwg.data_ptr(), nkw);
    dw_finalize_kernel<<<blocks, 256, 0, stream>>>((const float*)ws1.data_ptr(), (bf16*)dwu.data_ptr(), nkw);

    return {dx, dwg, dwu};
}

std::vector<torch::Tensor> linear_backward(torch::Tensor dout, torch::Tensor x, torch::Tensor w,
                                           int64_t cfg_nn, int64_t cfg_nt) {
    TORCH_CHECK(dout.is_cuda() && dout.dtype() == torch::kBFloat16 && dout.is_contiguous());
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous());
    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto dx = torch::empty({M, K}, x.options());
    nn_launch<1>((int)cfg_nn, (const bf16*)dout.data_ptr(), nullptr,
                 (const bf16*)w.data_ptr(), nullptr, (bf16*)dx.data_ptr(), M, N, K, stream);

    auto dw = torch::empty({N, K}, x.options());
    auto ws = torch::zeros({N, K}, x.options().dtype(torch::kFloat));
    nt_launch<1>((int)cfg_nt, (const bf16*)dout.data_ptr(), nullptr, (const bf16*)x.data_ptr(),
                 (float*)ws.data_ptr(), nullptr, M, N, K, stream);
    long nkw = (long)N * K;
    dw_finalize_kernel<<<(int)((nkw + 255) / 256), 256, 0, stream>>>(
        (const float*)ws.data_ptr(), (bf16*)dw.data_ptr(), nkw);

    return {dx, dw};
}

std::vector<torch::Tensor> linear_resid_rmsnorm_backward(torch::Tensor dy, torch::Tensor y,
                                                         torch::Tensor rstd, torch::Tensor x,
                                                         torch::Tensor w, int64_t cfg_nn, int64_t cfg_nt) {
    TORCH_CHECK(dy.is_cuda() && dy.dtype() == torch::kBFloat16 && dy.is_contiguous());
    TORCH_CHECK(y.is_contiguous() && x.is_contiguous() && w.is_contiguous());
    int M = x.size(0), K = x.size(1), N = w.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    auto dh = torch::empty({M, N}, dy.options());
    int warps_per_block = 8;
    int blocks = (M + warps_per_block - 1) / warps_per_block;
    rmsnorm_bwd_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        (const bf16*)dy.data_ptr(), (const bf16*)y.data_ptr(), (const float*)rstd.data_ptr(),
        (bf16*)dh.data_ptr(), M, N);

    auto dx = torch::empty({M, K}, x.options());
    nn_launch<1>((int)cfg_nn, (const bf16*)dh.data_ptr(), nullptr,
                 (const bf16*)w.data_ptr(), nullptr, (bf16*)dx.data_ptr(), M, N, K, stream);

    auto dw = torch::empty({N, K}, x.options());
    auto ws = torch::zeros({N, K}, x.options().dtype(torch::kFloat));
    nt_launch<1>((int)cfg_nt, (const bf16*)dh.data_ptr(), nullptr, (const bf16*)x.data_ptr(),
                 (float*)ws.data_ptr(), nullptr, M, N, K, stream);
    long nkw = (long)N * K;
    dw_finalize_kernel<<<(int)((nkw + 255) / 256), 256, 0, stream>>>(
        (const float*)ws.data_ptr(), (bf16*)dw.data_ptr(), nkw);

    return {dx, dw, dh};
}

torch::Tensor dbg_nn(torch::Tensor dy, torch::Tensor w, int64_t cfg) {
    int M = dy.size(0), N = dy.size(1), K = w.size(1);
    auto dx = torch::empty({M, K}, dy.options());
    nn_launch<1>((int)cfg, (const bf16*)dy.data_ptr(), nullptr, (const bf16*)w.data_ptr(),
                 nullptr, (bf16*)dx.data_ptr(), M, N, K, at::cuda::getCurrentCUDAStream());
    return dx;
}

torch::Tensor dbg_nt(torch::Tensor dy, torch::Tensor x, int64_t cfg) {
    int M = dy.size(0), N = dy.size(1), K = x.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto ws = torch::zeros({N, K}, dy.options().dtype(torch::kFloat));
    auto dw = torch::empty({N, K}, dy.options());
    nt_launch<1>((int)cfg, (const bf16*)dy.data_ptr(), nullptr, (const bf16*)x.data_ptr(),
                 (float*)ws.data_ptr(), nullptr, M, N, K, stream);
    long nkw = (long)N * K;
    dw_finalize_kernel<<<(int)((nkw + 255) / 256), 256, 0, stream>>>(
        (const float*)ws.data_ptr(), (bf16*)dw.data_ptr(), nkw);
    return dw;
}
