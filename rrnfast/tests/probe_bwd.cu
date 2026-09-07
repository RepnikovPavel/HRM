#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../csrc/edge_bwd.cu"

// pass-A replica incl. nt_accum + nt_flush: ws4 = dW4, ws3 = dW3 partial sums
__global__ void __launch_bounds__(RRN_THREADS, 1) probe_bwd_nt(
    const bf16* __restrict__ hw12, const int* __restrict__ nb, const float* __restrict__ b1,
    const bf16* __restrict__ W2, const bf16* __restrict__ b2,
    const bf16* __restrict__ W3, const bf16* __restrict__ b3,
    const bf16* __restrict__ W4, const bf16* __restrict__ b4,
    const bf16* __restrict__ dm,
    float* __restrict__ ws4, float* __restrict__ ws3)
{
    extern __shared__ bf16 smem[];
    bf16* T0 = smem;
    bf16* T1 = smem + RRN_TILE_ELEMS;
    bf16* T2 = smem + 2 * RRN_TILE_ELEMS;

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int m0 = (warp >> 1) * 16, nh = (warp & 1) * 48;
    const int b = 0, j0 = blockIdx.x * RRN_NB;
    float acc[6][4];

    build_e0(T0, hw12, nb, b1, b, j0, tid);
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 6; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
    mv_tile(smem_u32(T0), m0, W2, nh, lane, acc);
    layer_epilogue_sts(acc, b2, T1, m0, nh, lane);
    __syncthreads();
#pragma unroll
    for (int nt = 0; nt < 6; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
    mv_tile(smem_u32(T1), m0, W3, nh, lane, acc);
    layer_epilogue_sts(acc, b3, T0, m0, nh, lane);

    float de3[6][4];
    make_de3(de3, dm, b, j0, m0, nh, lane, 0, 0.f, 0.f, 0, 0);
    __syncthreads();
    sts_acc(T2, de3, m0, nh, lane);
    __syncthreads();

    float acc4[8][4], acc3[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc4[t][q] = acc3[t][q] = 0.f;

    nt_accum(smem_u32(T2), smem_u32(T0), warp, lane, acc4);

#pragma unroll
    for (int nt = 0; nt < 6; ++nt)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
    mv_tile(smem_u32(T2), m0, W4, nh, lane, acc);
    relu_mask_tile(acc, T0, m0, nh, lane);
    __syncthreads();
    sts_acc(T2, acc, m0, nh, lane);
    __syncthreads();

    nt_accum(smem_u32(T2), smem_u32(T1), warp, lane, acc3);

    nt_flush(ws4, warp, lane, acc4);
    nt_flush(ws3, warp, lane, acc3);
}

std::vector<torch::Tensor> run_probe(torch::Tensor hw12, torch::Tensor nb, torch::Tensor b1,
                                     torch::Tensor w2, torch::Tensor b2, torch::Tensor w3,
                                     torch::Tensor b3, torch::Tensor w4, torch::Tensor b4,
                                     torch::Tensor dm) {
    auto f32 = torch::TensorOptions().dtype(torch::kFloat).device(hw12.device());
    auto ws4 = torch::zeros({96, 96}, f32);
    auto ws3 = torch::zeros({96, 96}, f32);
    int smem = 3 * RRN_TILE_ELEMS * 2;
    cudaFuncSetAttribute(probe_bwd_nt, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    probe_bwd_nt<<<21, RRN_THREADS, smem>>>(
        (const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(), (const float*)b1.data_ptr(),
        (const bf16*)w2.data_ptr(), (const bf16*)b2.data_ptr(),
        (const bf16*)w3.data_ptr(), (const bf16*)b3.data_ptr(),
        (const bf16*)w4.data_ptr(), (const bf16*)b4.data_ptr(),
        (const bf16*)dm.data_ptr(), (float*)ws4.data_ptr(), (float*)ws3.data_ptr());
    return {ws4, ws3};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
