#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../csrc/edge_bwd.cu"

__device__ __forceinline__ void mv_tile_const(uint32_t tile_base, int m0, const bf16* w,
                                              int n0, int lane, float acc[6][4]) {
    uint32_t af[6][4];
#pragma unroll
    for (int s = 0; s < 6; ++s)
        ld_tile_afrag(tile_base, m0, s * 16, lane, af[s][0], af[s][1], af[s][2], af[s][3]);
#pragma unroll
    for (int s = 0; s < 6; ++s)
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
            mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                     af[s][0], af[s][1], af[s][2], af[s][3], 0x3f803f80u, 0x3f803f80u);
}


// LEVEL: 0=e0 only, 1=+L2, 2=+L3, 3=full (+dropout off +rowsum+m write)
template <int LEVEL>
__global__ void __launch_bounds__(RRN_THREADS, 1) probe_fwd(
    const bf16* __restrict__ hw12, const int* __restrict__ nb, const float* __restrict__ b1,
    const bf16* __restrict__ W2, const bf16* __restrict__ b2,
    const bf16* __restrict__ W3, const bf16* __restrict__ b3,
    const bf16* __restrict__ W4, const bf16* __restrict__ b4,
    bf16* __restrict__ m_out)
{
    __shared__ bf16 tileA[RRN_TILE_ELEMS];
    __shared__ bf16 tileB[RRN_TILE_ELEMS];
    __shared__ float macc[RRN_NB][RRN_D];

    const int b = blockIdx.y;
    const int j0 = blockIdx.x * RRN_NB;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int m0 = (warp >> 1) * 16, nh = (warp & 1) * 48;

    build_e0(tileA, hw12, nb, b1, b, j0, tid);
    for (int idx = tid; idx < RRN_NB * RRN_D; idx += RRN_THREADS)
        macc[idx / RRN_D][idx % RRN_D] = 0.f;
    __syncthreads();

    float acc[6][4];
    bf16* tiles[2] = {tileA, tileB};
    const bf16* wl[3] = {W2, W3, W4};
    const bf16* bl[3] = {b2, b3, b4};
    int cur = 0;
#pragma unroll
    for (int l = 0; l < 3; ++l) {
        if (l >= LEVEL) break;
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        if (LEVEL < 10) mv_tile(smem_u32(tiles[cur]), m0, wl[l], nh, lane, acc);
        else mv_tile_const(smem_u32(tiles[cur]), m0, wl[l], nh, lane, acc);
        bool last = (l == 2);
#pragma unroll
        for (int nt = 0; nt < 6; ++nt) {
            int c = nh + nt * 8 + (lane & 3) * 2;
            __nv_bfloat162 bb = *(const __nv_bfloat162*)(bl[l] + c);
            float f0 = __bfloat162float(bb.x), f1 = __bfloat162float(bb.y);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float v0 = acc[nt][h * 2] + f0;
                float v1 = acc[nt][h * 2 + 1] + f1;
                if (!last) {
                    __nv_bfloat162 pk = __float22bfloat162_rn(
                        make_float2(fmaxf(v0, 0.f), fmaxf(v1, 0.f)));
                    int r = m0 + (lane >> 2) + h * 8;
                    *(__nv_bfloat162*)&tiles[cur ^ 1][r * RRN_ASTR + c] = pk;
                } else {
                    acc[nt][h * 2] = bf16_round(v0);
                    acc[nt][h * 2 + 1] = bf16_round(v1);
                }
            }
        }
        if (!last) {
            __syncthreads();
            cur ^= 1;
        }
    }
    if (LEVEL == 0) {
        // drain e0 through the same rowsum path to keep it honest
        for (int idx = tid; idx < RRN_ROWS * RRN_D / 2; idx += RRN_THREADS) {
            int r = idx / (RRN_D / 2), cp = idx % (RRN_D / 2);
            __nv_bfloat162 v = *(__nv_bfloat162*)&tileA[r * RRN_ASTR + cp * 2];
            atomicAdd(&macc[r / RRN_E][cp * 2], __bfloat162float(v.x));
        }
    }
    if (LEVEL == 3) {
        rowsum_node(acc, m0, lane, macc, nh);
    }
    __syncthreads();
    for (int idx = tid; idx < RRN_NB * RRN_D; idx += RRN_THREADS) {
        int l = idx / RRN_D, c = idx % RRN_D;
        int j = j0 + l;
        if (j < RRN_NN)
            m_out[(size_t)(b * RRN_NN + j) * RRN_D + c] = __float2bfloat16(macc[l][c]);
    }
}

void run_probe(torch::Tensor hw12, torch::Tensor nb, torch::Tensor b1,
               torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3,
               torch::Tensor w4, torch::Tensor b4, torch::Tensor m, int64_t B, int64_t level) {
    dim3 grid(21, (int)B);
    auto s = at::cuda::getCurrentCUDAStream();
#define A(i) (const bf16*)w##i.data_ptr(), (const bf16*)b##i.data_ptr()
    if (level == 0)
        probe_fwd<0><<<grid, RRN_THREADS, 0, s>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                                  (const float*)b1.data_ptr(), A(2), A(3), A(4),
                                                  (bf16*)m.data_ptr());
    if (level == 1)
        probe_fwd<1><<<grid, RRN_THREADS, 0, s>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                                  (const float*)b1.data_ptr(), A(2), A(3), A(4),
                                                  (bf16*)m.data_ptr());
    if (level == 2)
        probe_fwd<2><<<grid, RRN_THREADS, 0, s>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                                  (const float*)b1.data_ptr(), A(2), A(3), A(4),
                                                  (bf16*)m.data_ptr());
    if (level == 3 || level == 13)
        probe_fwd<3><<<grid, RRN_THREADS, 0, s>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                                  (const float*)b1.data_ptr(), A(2), A(3), A(4),
                                                  (bf16*)m.data_ptr());
    if (level == 13)
        probe_fwd<13><<<grid, RRN_THREADS, 0, s>>>((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                                                   (const float*)b1.data_ptr(), A(2), A(3), A(4),
                                                   (bf16*)m.data_ptr());
#undef A
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
