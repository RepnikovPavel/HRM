#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "edge_common.cuh"

// 5 warps x 160 threads: warp = one m16 tile over the full 96-wide output;
// a 168-reg block of 160 threads allows 2 blocks/SM, so one block's
// syncthreads/gather stalls overlap with the other block's mma
#define RRN_FWD_THREADS 160

__global__ void __launch_bounds__(RRN_FWD_THREADS, 2) edge_fwd_kernel(
    const bf16* __restrict__ hw12, const int* __restrict__ nb,
    const float* __restrict__ b1,
    const bf16* __restrict__ W2, const bf16* __restrict__ b2,
    const bf16* __restrict__ W3, const bf16* __restrict__ b3,
    const bf16* __restrict__ W4, const bf16* __restrict__ b4,
    bf16* __restrict__ m_out,
    int train, float p, float scale, uint64_t seed, int step)
{
    __shared__ bf16 tileA[RRN_TILE_ELEMS];
    __shared__ bf16 tileB[RRN_TILE_ELEMS];
    __shared__ float macc[RRN_NB][RRN_D];

    const int b = blockIdx.y;
    const int j0 = blockIdx.x * RRN_NB;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int m0 = warp * 16;

    build_e0(tileA, hw12, nb, b1, b, j0, tid);
    for (int idx = tid; idx < RRN_NB * RRN_D; idx += RRN_FWD_THREADS)
        macc[idx / RRN_D][idx % RRN_D] = 0.f;
    __syncthreads();

    const bf16* wl[3] = {W2, W3, W4};
    const bf16* bl[3] = {b2, b3, b4};
    bf16* tiles[2] = {tileA, tileB};
    int cur = 0;
    float acc[12][4];
#pragma unroll
    for (int l = 0; l < 3; ++l) {
#pragma unroll
        for (int nt = 0; nt < 12; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<12>(smem_u32(tiles[cur]), m0, wl[l], 0, lane, acc);
        bool last = (l == 2);
#pragma unroll
        for (int nt = 0; nt < 12; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
            __nv_bfloat162 bb = *(const __nv_bfloat162*)(bl[l] + c);
            float f0 = __bfloat162float(bb.x), f1 = __bfloat162float(bb.y);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                float v0 = acc[nt][h * 2] + f0;
                float v1 = acc[nt][h * 2 + 1] + f1;
                if (!last) {
                    v0 = fmaxf(v0, 0.f);
                    v1 = fmaxf(v1, 0.f);
                    __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(v0, v1));
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

    if (train) {
#pragma unroll
        for (int nt = 0; nt < 12; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = m0 + (lane >> 2) + h * 8;
                uint edge = (uint)((b * RRN_NN + j0 + r / RRN_E) * RRN_E + r % RRN_E);
                drop_pair(seed, step, edge, c, p, scale, acc[nt][h * 2], acc[nt][h * 2 + 1]);
            }
        }
    }

    rowsum_node<12>(acc, m0, lane, macc, 0);
    __syncthreads();
    for (int idx = tid; idx < RRN_NB * RRN_D; idx += RRN_FWD_THREADS) {
        int l = idx / RRN_D, c = idx % RRN_D;
        int j = j0 + l;
        if (j < RRN_NN)
            m_out[(size_t)(b * RRN_NN + j) * RRN_D + c] = __float2bfloat16(macc[l][c]);
    }
}

void edge_fwd_launch(const bf16* hw12, const int* nb, const float* b1,
                     const bf16* W2, const bf16* b2, const bf16* W3, const bf16* b3,
                     const bf16* W4, const bf16* b4, bf16* m_out,
                     int B, int train, float p, float scale, uint64_t seed, int step,
                     cudaStream_t stream) {
    dim3 grid((RRN_NN + RRN_NB - 1) / RRN_NB, B);
    edge_fwd_kernel<<<grid, RRN_FWD_THREADS, 0, stream>>>(hw12, nb, b1, W2, b2, W3, b3, W4, b4,
                                                          m_out, train, p, scale, seed, step);
}
