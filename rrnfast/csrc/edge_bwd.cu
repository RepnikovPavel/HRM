#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "edge_common.cuh"

// dW accumulators live in registers across the persistent loop and are flushed
// once per block; per-block atomic flush (216 blocks x 9216 floats) is what
// makes the fp32 workspace reduction cheap enough
__device__ __forceinline__ void nt_accum(uint32_t baseA, uint32_t baseB, int warp, int lane,
                                         float accw[8][4]) {
    int ti = 0;
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        int tile = warp + t * RRN_WARPS;
        if (tile >= 72) break;
        int mt = tile / 12, nt2 = tile % 12;
#pragma unroll
        for (int ks = 0; ks < 5; ++ks) {
            uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
            ld_tile_tfrag_a(baseA, mt * 16, ks * 16, lane, a0, a1, a2, a3);
            ld_tile_tfrag_b(baseB, (nt2 & ~1) * 8, ks * 16, lane, b0, b1, b2, b3);
            if (nt2 & 1)
                mma_bf16(accw[ti][0], accw[ti][1], accw[ti][2], accw[ti][3], a0, a1, a2, a3, b2, b3);
            else
                mma_bf16(accw[ti][0], accw[ti][1], accw[ti][2], accw[ti][3], a0, a1, a2, a3, b0, b1);
        }
        ++ti;
    }
}

__device__ __forceinline__ void nt_flush(float* ws, int warp, int lane, const float accw[8][4]) {
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        int tile = warp + t * RRN_WARPS;
        if (tile >= 72) break;
        int mt = tile / 12, nt2 = tile % 12;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int o = mt * 16 + (lane >> 2) + h * 8;
            int i = nt2 * 8 + (lane & 3) * 2;
            atomicAdd(ws + o * RRN_D + i, accw[t][h * 2]);
            atomicAdd(ws + o * RRN_D + i + 1, accw[t][h * 2 + 1]);
        }
    }
}

template <int NNT>
__device__ __forceinline__ void colsum_db(const float (*v)[4], int lane, int nh, float* db) {
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
        float s0 = v[nt][0] + v[nt][2];
        float s1 = v[nt][1] + v[nt][3];
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            s0 += __shfl_xor_sync(0xffffffffu, s0, off);
            s1 += __shfl_xor_sync(0xffffffffu, s1, off);
        }
        if (lane < 4) {
            atomicAdd(db + c, s0);
            atomicAdd(db + c + 1, s1);
        }
    }
}


__device__ __forceinline__ void layer_epilogue_sts(float acc[6][4], const bf16* bias,
                                                   bf16* out, int m0, int nh, int lane) {
#pragma unroll
    for (int nt = 0; nt < 6; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
        __nv_bfloat162 bb = *(const __nv_bfloat162*)(bias + c);
        float f0 = __bfloat162float(bb.x), f1 = __bfloat162float(bb.y);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            __nv_bfloat162 pk = __float22bfloat162_rn(
                make_float2(fmaxf(acc[nt][h * 2] + f0, 0.f), fmaxf(acc[nt][h * 2 + 1] + f1, 0.f)));
            int r = m0 + (lane >> 2) + h * 8;
            *(__nv_bfloat162*)&out[r * RRN_ASTR + c] = pk;
        }
    }
}

// de3[r, c] = dm[b, j(r), c] * drop_mask(b, j(r), k(r), c); rounded to bf16 and
// stored back as float so downstream mma sees exactly the reference's values
__device__ __forceinline__ void make_de3(float de3[6][4], const bf16* dm, int b, int j0,
                                         int m0, int nh, int lane,
                                         int train, float p, float scale, uint64_t seed, int step) {
#pragma unroll
    for (int nt = 0; nt < 6; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int r = m0 + (lane >> 2) + h * 8;
            int l = r / RRN_E, k = r % RRN_E;
            int j = j0 + l;
            float d0 = 0.f, d1 = 0.f;
            if (j < RRN_NN) {
                __nv_bfloat162 dm2 = *(const __nv_bfloat162*)(dm + (size_t)(b * RRN_NN + j) * 192 + c);
                d0 = __bfloat162float(dm2.x);
                d1 = __bfloat162float(dm2.y);
            }
            if (train) {
                uint edge = (uint)((b * RRN_NN + j) * RRN_E + k);
                drop_pair(seed, step, edge, c, p, scale, d0, d1);
            }
            de3[nt][h * 2] = d0;
            de3[nt][h * 2 + 1] = d1;
        }
    }
}

__device__ __forceinline__ void sts_acc(bf16* tile, const float v[6][4], int m0, int nh, int lane) {
#pragma unroll
    for (int nt = 0; nt < 6; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int r = m0 + (lane >> 2) + h * 8;
            __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(v[nt][h * 2], v[nt][h * 2 + 1]));
            *(__nv_bfloat162*)&tile[r * RRN_ASTR + c] = pk;
        }
    }
}

// da = round(round(mv) * (prev > 0)); prev mask read from the smem tile that
// already holds the rounded forward activation
__device__ __forceinline__ void relu_mask_tile(float acc[6][4], const bf16* prev,
                                               int m0, int nh, int lane) {
#pragma unroll
    for (int nt = 0; nt < 6; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int r = m0 + (lane >> 2) + h * 8;
            __nv_bfloat162 pv = *(const __nv_bfloat162*)&prev[r * RRN_ASTR + c];
            acc[nt][h * 2] = bf16_round(bf16_round(acc[nt][h * 2]) * (__bfloat162float(pv.x) > 0.f ? 1.f : 0.f));
            acc[nt][h * 2 + 1] = bf16_round(bf16_round(acc[nt][h * 2 + 1]) * (__bfloat162float(pv.y) > 0.f ? 1.f : 0.f));
        }
    }
}

__global__ void __launch_bounds__(RRN_THREADS, 1) edge_bwd_a_kernel(
    const bf16* __restrict__ hw12, const int* __restrict__ nb, const float* __restrict__ b1,
    const bf16* __restrict__ W2, const bf16* __restrict__ b2,
    const bf16* __restrict__ W3, const bf16* __restrict__ b3,
    const bf16* __restrict__ W4, const bf16* __restrict__ b4,
    const bf16* __restrict__ W4t,
    const bf16* __restrict__ dm,
    float* __restrict__ ws4, float* __restrict__ ws3,
    float* __restrict__ gdb4, float* __restrict__ gdb3,
    int B, int total_chunks, int train, float p, float scale, uint64_t seed, int step)
{
    extern __shared__ bf16 smem[];
    bf16* T0 = smem;
    bf16* T1 = smem + RRN_TILE_ELEMS;
    bf16* T2 = smem + 2 * RRN_TILE_ELEMS;
    float* dbs = (float*)(smem + 3 * RRN_TILE_ELEMS);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int m0 = (warp >> 1) * 16, nh = (warp & 1) * 48;

    float acc4[8][4], acc3[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc4[t][q] = acc3[t][q] = 0.f;
    for (int idx = tid; idx < 2 * RRN_D; idx += RRN_THREADS) dbs[idx] = 0.f;
    __syncthreads();

    for (int chunk = blockIdx.x; chunk < total_chunks; chunk += gridDim.x) {
        int b = chunk / 21, j0 = (chunk % 21) * RRN_NB;
        float acc[6][4];

        build_e0(T0, hw12, nb, b1, b, j0, tid);
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T0), m0, W2, nh, lane, acc);
        layer_epilogue_sts(acc, b2, T1, m0, nh, lane);
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T1), m0, W3, nh, lane, acc);
        layer_epilogue_sts(acc, b3, T0, m0, nh, lane);

        float de3[6][4];
        make_de3(de3, dm, b, j0, m0, nh, lane, train, p, scale, seed, step);
        colsum_db<6>(de3, lane, nh, dbs);
        __syncthreads();
        sts_acc(T2, de3, m0, nh, lane);
        __syncthreads();

        nt_accum(smem_u32(T2), smem_u32(T0), warp, lane, acc4);

#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T2), m0, W4t, nh, lane, acc);
        relu_mask_tile(acc, T0, m0, nh, lane);
        colsum_db<6>(acc, lane, nh, dbs + RRN_D);
        __syncthreads();
        sts_acc(T2, acc, m0, nh, lane);
        __syncthreads();

        nt_accum(smem_u32(T2), smem_u32(T1), warp, lane, acc3);
        __syncthreads();
    }

    nt_flush(ws4, warp, lane, acc4);
    nt_flush(ws3, warp, lane, acc3);
    for (int idx = tid; idx < RRN_D; idx += RRN_THREADS) {
        atomicAdd(gdb4 + idx, dbs[idx]);
        atomicAdd(gdb3 + idx, dbs[RRN_D + idx]);
    }
}

__global__ void __launch_bounds__(RRN_THREADS, 1) edge_bwd_b_kernel(
    const bf16* __restrict__ hw12, const int* __restrict__ nb, const float* __restrict__ b1,
    const bf16* __restrict__ W2, const bf16* __restrict__ b2,
    const bf16* __restrict__ W3, const bf16* __restrict__ b3,
    const bf16* __restrict__ W4, const bf16* __restrict__ b4,
    const bf16* __restrict__ W4t, const bf16* __restrict__ W3t, const bf16* __restrict__ W2t,
    const bf16* __restrict__ dm,
    float* __restrict__ ws2, float* __restrict__ gdb2, float* __restrict__ gdb1,
    bf16* __restrict__ dz0,
    int B, int total_chunks, int train, float p, float scale, uint64_t seed, int step)
{
    extern __shared__ bf16 smem[];
    bf16* T0 = smem;                       // e0
    bf16* T1 = smem + RRN_TILE_ELEMS;      // e1
    bf16* T2 = smem + 2 * RRN_TILE_ELEMS;  // da0 then dz0
    float* dbs = (float*)(smem + 3 * RRN_TILE_ELEMS);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int m0 = (warp >> 1) * 16, nh = (warp & 1) * 48;

    float acc2[8][4];
#pragma unroll
    for (int t = 0; t < 8; ++t)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc2[t][q] = 0.f;
    for (int idx = tid; idx < 2 * RRN_D; idx += RRN_THREADS) dbs[idx] = 0.f;
    __syncthreads();

    for (int chunk = blockIdx.x; chunk < total_chunks; chunk += gridDim.x) {
        int b = chunk / 21, j0 = (chunk % 21) * RRN_NB;
        float acc[6][4];

        build_e0(T0, hw12, nb, b1, b, j0, tid);
        __syncthreads();
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T0), m0, W2, nh, lane, acc);
        layer_epilogue_sts(acc, b2, T1, m0, nh, lane);
        __syncthreads();

        // e2 stays in registers: the da1 mask reads the same (row, col) slots
        float e2[6][4];
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) e2[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T1), m0, W3, nh, lane, e2);
#pragma unroll
        for (int nt = 0; nt < 6; ++nt) {
            int c = nh + nt * 8 + (lane & 3) * 2;
            __nv_bfloat162 bb = *(const __nv_bfloat162*)(b3 + c);
            float f0 = __bfloat162float(bb.x), f1 = __bfloat162float(bb.y);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                e2[nt][h * 2] = bf16_round(fmaxf(e2[nt][h * 2] + f0, 0.f));
                e2[nt][h * 2 + 1] = bf16_round(fmaxf(e2[nt][h * 2 + 1] + f1, 0.f));
            }
        }

        float de3[6][4];
        make_de3(de3, dm, b, j0, m0, nh, lane, train, p, scale, seed, step);
        sts_acc(T2, de3, m0, nh, lane);
        __syncthreads();

#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T2), m0, W4t, nh, lane, acc);
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q)
                acc[nt][q] = bf16_round(bf16_round(acc[nt][q]) * (e2[nt][q] > 0.f ? 1.f : 0.f));
        __syncthreads();
        sts_acc(T2, acc, m0, nh, lane);
        __syncthreads();

        float de1[6][4];
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) de1[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T2), m0, W3t, nh, lane, de1);
        relu_mask_tile(de1, T1, m0, nh, lane);
        colsum_db<6>(de1, lane, nh, dbs);
        __syncthreads();
        sts_acc(T2, de1, m0, nh, lane);
        __syncthreads();

        nt_accum(smem_u32(T2), smem_u32(T0), warp, lane, acc2);

        float de0[6][4];
#pragma unroll
        for (int nt = 0; nt < 6; ++nt)
#pragma unroll
            for (int q = 0; q < 4; ++q) de0[nt][q] = 0.f;
        mv_tile<6>(smem_u32(T2), m0, W2t, nh, lane, de0);
        relu_mask_tile(de0, T0, m0, nh, lane);
        colsum_db<6>(de0, lane, nh, dbs + RRN_D);
        __syncthreads();
        sts_acc(T2, de0, m0, nh, lane);
        __syncthreads();

        for (int idx = tid; idx < RRN_ROWS * 12; idx += RRN_THREADS) {
            int r = idx / 12, cb = idx % 12;
            if (j0 + r / RRN_E < RRN_NN) {
                uint4 v = *(uint4*)&T2[r * RRN_ASTR + cb * 8];
                *(uint4*)(dz0 + ((size_t)(b * RRN_NN + j0) * RRN_E) * RRN_D + r * RRN_D + cb * 8) = v;
            }
        }
        __syncthreads();
    }

    nt_flush(ws2, warp, lane, acc2);
    for (int idx = tid; idx < RRN_D; idx += RRN_THREADS) {
        atomicAdd(gdb2 + idx, dbs[idx]);
        atomicAdd(gdb1 + idx, dbs[RRN_D + idx]);
    }
}

// dhw12[b, s, :96] = sum_k dz0[b, nb[s,k], pos[s,k], :] (grad wrt hw1, graph is
// symmetric so predecessors of s are exactly its neighbors);
// dhw12[b, s, 96:] = sum_k dz0[b, s, k, :] (grad wrt hw2)
__global__ void dhw_kernel(const bf16* __restrict__ dz0, const int* __restrict__ nb,
                           const int* __restrict__ pos, bf16* __restrict__ dhw12) {
    int bj = blockIdx.x;
    int b = bj / RRN_NN, j = bj % RRN_NN;
    int c = threadIdx.x;
    float a1 = 0.f, a2 = 0.f;
    const bf16* own = dz0 + (size_t)bj * RRN_E * RRN_D;
#pragma unroll 4
    for (int k = 0; k < RRN_E; ++k) {
        a2 += __bfloat162float(own[k * RRN_D + c]);
        int src = nb[j * RRN_E + k];
        int pp = pos[j * RRN_E + k];
        a1 += __bfloat162float(dz0[((size_t)(b * RRN_NN + src) * RRN_E + pp) * RRN_D + c]);
    }
    dhw12[(size_t)bj * 192 + c] = __float2bfloat16(a1);
    dhw12[(size_t)bj * 192 + 96 + c] = __float2bfloat16(a2);
}

void edge_bwd_launch(const bf16* hw12, const int* nb, const int* pos, const float* b1,
                     const bf16* W2, const bf16* b2, const bf16* W3, const bf16* b3,
                     const bf16* W4, const bf16* b4,
                     const bf16* W2t, const bf16* W3t, const bf16* W4t, const bf16* dm,
                     float* ws2, float* ws3, float* ws4,
                     float* db1, float* db2, float* db3, float* db4,
                     bf16* dz0, bf16* dhw12,
                     int B, int train, float p, float scale, uint64_t seed, int step,
                     int sm_count, cudaStream_t stream) {
    int total_chunks = ((RRN_NN + RRN_NB - 1) / RRN_NB) * B;
    int smem = 3 * RRN_TILE_ELEMS * 2 + 2 * RRN_D * 4;
    static bool cfgd = false;
    if (!cfgd) {
        cudaFuncSetAttribute(edge_bwd_a_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(edge_bwd_b_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cfgd = true;
    }
    int blocks = 2 * sm_count;
    edge_bwd_a_kernel<<<blocks, RRN_THREADS, smem, stream>>>(
        hw12, nb, b1, W2, b2, W3, b3, W4, b4, W4t, dm, ws4, ws3, db4, db3,
        B, total_chunks, train, p, scale, seed, step);
    edge_bwd_b_kernel<<<blocks, RRN_THREADS, smem, stream>>>(
        hw12, nb, b1, W2, b2, W3, b3, W4, b4, W4t, W3t, W2t, dm, ws2, db2, db1, dz0,
        B, total_chunks, train, p, scale, seed, step);
    dhw_kernel<<<B * RRN_NN, RRN_D, 0, stream>>>(dz0, nb, pos, dhw12);
}
