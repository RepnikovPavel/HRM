#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../csrc/edge_common.cuh"

__global__ void probe_e0(const bf16* hw12, const int* nb, const float* b1, bf16* out) {
    __shared__ bf16 tile[RRN_TILE_ELEMS];
    int b = blockIdx.y, j0 = blockIdx.x * RRN_NB;
    // same body as build_e0 (copy to test device function)
    for (int idx = threadIdx.x; idx < RRN_ROWS * 12; idx += RRN_THREADS) {
        int r = idx / 12, cb = idx % 12;
        int l = r / RRN_E, k = r % RRN_E;
        int j = j0 + l;
        bool v = j < RRN_NN;
        int src = v ? nb[j * RRN_E + k] : 0;
        int jr = v ? j : 0;
        uint4 v1 = *(const uint4*)(hw12 + (size_t)(b * RRN_NN + src) * 192 + cb * 8);
        uint4 v2 = *(const uint4*)(hw12 + (size_t)(b * RRN_NN + jr) * 192 + 96 + cb * 8);
        uint4 vb = *(const uint4*)(b1 + cb * 8);
        const bf16* p1 = (const bf16*)&v1;
        const bf16* p2 = (const bf16*)&v2;
        const float* pb = (const float*)&vb;
        uint4 o;
        bf16* po = (bf16*)&o;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            po[i] = __float2bfloat16(fmaxf(__bfloat162float(p1[i]) + __bfloat162float(p2[i]) + pb[i], 0.f));
        *(uint4*)&tile[r * RRN_ASTR + cb * 8] = o;
    }
    __syncthreads();
    // write out [80, 96] compact
    for (int idx = threadIdx.x; idx < RRN_ROWS * 12; idx += RRN_THREADS) {
        int r = idx / 12, cb = idx % 12;
        int j = j0 + r / RRN_E, k = r % RRN_E;
        if (j < RRN_NN)
            *(uint4*)(out + ((size_t)(b * RRN_NN + j) * RRN_E + k) * RRN_D + cb * 8) =
                *(uint4*)&tile[r * RRN_ASTR + cb * 8];
    }
}

torch::Tensor run_probe(torch::Tensor hw12, torch::Tensor nb, torch::Tensor b1, int64_t B) {
    auto out = torch::zeros({B, 81, 20, 96}, hw12.options());
    dim3 grid(21, B);
    probe_e0<<<grid, RRN_THREADS>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                    (const float*)b1.data_ptr(), (bf16*)out.data_ptr());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
