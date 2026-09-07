#include <torch/extension.h>
#include "../csrc/edge_common.cuh"

// fills tile[r][c] = 100*r + c, loads afrag at (m0=0, k0), each thread stores
// its 8 elements as (row, col, value) triplets
__global__ void probe(float* out, int m0, int k0) {
    __shared__ bf16 tile[RRN_TILE_ELEMS];
    int tid = threadIdx.x;
    for (int i = tid; i < RRN_ROWS * RRN_D; i += blockDim.x) {
        int r = i / RRN_D, c = i % RRN_D;
        tile[r * RRN_ASTR + c] = __float2bfloat16((float)(100 * r + c));
    }
    __syncthreads();
    if (tid >= 32) return;
    int lane = tid;
    uint32_t a0, a1, a2, a3;
    ld_tile_afrag(smem_u32(tile), m0, k0, lane, a0, a1, a2, a3);
    uint32_t ar[4] = {a0, a1, a2, a3};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
        __nv_bfloat162 v = *(__nv_bfloat162*)&ar[q];
        out[(lane * 4 + q) * 2] = __bfloat162float(v.x);
        out[(lane * 4 + q) * 2 + 1] = __bfloat162float(v.y);
    }
}

torch::Tensor run_probe(int64_t m0, int64_t k0) {
    auto out = torch::zeros({32, 8}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    probe<<<1, 64>>>((float*)out.data_ptr(), (int)m0, (int)k0);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
