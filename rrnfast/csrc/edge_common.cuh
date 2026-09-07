#pragma once
#include <cuda_bf16.h>
#include <cstdint>
#include "gemm_family.cuh"

// one threadblock (320 thr = 10 warps) handles RRN_NB sudoku nodes of one batch
// element; edge rows (80 = 4 nodes x 20 edges) map to 5 m16 mma tiles; warp w
// owns m-tile w>>1 and column half w&1 (48 of 96 output features)
#define RRN_D 96
#define RRN_E 20
#define RRN_NN 81
#define RRN_NB 4
#define RRN_ROWS 80
#define RRN_THREADS 320
#define RRN_WARPS 10
// 104 = 96 + 8 keeps every 16B segment of ldmatrix on distinct banks (208B row
// stride cycles through all 8 bank groups) and stays 16B-aligned for uint4 IO
#define RRN_ASTR 104
#define RRN_TILE_ELEMS (RRN_ROWS * RRN_ASTR)

__device__ __forceinline__ uint4 philox4x32_10(uint4 ctr, uint2 key) {
#pragma unroll
    for (int i = 0; i < 10; ++i) {
        uint64_t p0 = 0xD2511F53ull * ctr.x;
        uint64_t p1 = 0xCD9E8D57ull * ctr.z;
        ctr = make_uint4((uint)(p1 >> 32) ^ ctr.y ^ key.x, (uint)p1,
                         (uint)(p0 >> 32) ^ ctr.w ^ key.y, (uint)p0);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return ctr;
}

__device__ __forceinline__ float philox_uniform(uint v) {
    return (float)(v >> 8) * (1.0f / 16777216.0f);
}

// A-operand fragment (m16k16) from a row-major [80][RRN_ASTR] smem tile
__device__ __forceinline__ void ld_tile_afrag(uint32_t base, int m0, int k0, int lane,
                                              uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    uint32_t addr = base + (m0 + (lane & 15)) * (RRN_ASTR * 2) + (k0 + (lane >> 4) * 8) * 2;
    ldsm_x4(addr, a0, a1, a2, a3);
}

// transposed fragments from a [reduction][dim] smem tile (for dW += g^T @ a):
// A-operand over output dim m0, reduction offset k0
__device__ __forceinline__ void ld_tile_tfrag_a(uint32_t base, int m0, int k0, int lane,
                                                uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    uint32_t addr = base + (k0 + ((lane >> 4) << 3) + (lane & 7)) * (RRN_ASTR * 2)
                  + (m0 + (((lane >> 3) & 1) << 3)) * 2;
    ldsm_x4_trans(addr, a0, a1, a2, a3);
}

// B-operand pair (two n8 tiles) over output dim n0, reduction offset k0
__device__ __forceinline__ void ld_tile_tfrag_b(uint32_t base, int n0, int k0, int lane,
                                                uint32_t& b00, uint32_t& b01,
                                                uint32_t& b10, uint32_t& b11) {
    uint32_t addr = base + (k0 + (((lane >> 3) & 1) << 3) + (lane & 7)) * (RRN_ASTR * 2)
                  + (n0 + ((lane >> 4) << 3)) * 2;
    ldsm_x4_trans(addr, b00, b01, b10, b11);
}

// B-operand fragment straight from a row-major [96,96] weight in global (L1-resident):
// thread needs W[n][k + (lane%4)*2 + {0,1}] and +8, two aligned b32 loads
__device__ __forceinline__ void ld_wfrag(const bf16* w, int n, int k0, int lane,
                                         uint32_t& b0, uint32_t& b1) {
    const bf16* p = w + n * RRN_D + k0 + (lane & 3) * 2;
    b0 = *(const uint32_t*)p;
    b1 = *(const uint32_t*)(p + 8);
}

// acc[NNT][4] += tile[m0..m0+16] @ W^T[:, n0..n0+NNT*8]
template <int NNT>
__device__ __forceinline__ void mv_tile(uint32_t tile_base, int m0, const bf16* w,
                                        int n0, int lane, float (*acc)[4]) {
    uint32_t af[6][4];
#pragma unroll
    for (int s = 0; s < 6; ++s)
        ld_tile_afrag(tile_base, m0, s * 16, lane, af[s][0], af[s][1], af[s][2], af[s][3]);
#pragma unroll
    for (int s = 0; s < 6; ++s) {
#pragma unroll
        for (int nt = 0; nt < NNT; ++nt) {
            uint32_t b0, b1;
            ld_wfrag(w, n0 + nt * 8 + (lane >> 2), s * 16, lane, b0, b1);
            mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3],
                     af[s][0], af[s][1], af[s][2], af[s][3], b0, b1);
        }
    }
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// dropout mask on (b, j, k, feature): counter (edge_idx, feature/4, step)
__device__ __forceinline__ void drop_pair(uint64_t seed, int step, uint edge,
                                          int c, float p, float scale,
                                          float& v0, float& v1) {
    uint2 key = make_uint2((uint)seed, (uint)(seed >> 32));
    uint4 rnd = philox4x32_10(make_uint4(edge, (uint)(c >> 2), (uint)step, 0u), key);
    float u0 = ((c & 3) == 0) ? philox_uniform(rnd.x) : philox_uniform(rnd.z);
    float u1 = ((c & 3) == 0) ? philox_uniform(rnd.y) : philox_uniform(rnd.w);
    v0 = bf16_round(v0 * (u0 >= p ? scale : 0.f));
    v1 = bf16_round(v1 * (u1 >= p ? scale : 0.f));
}

// e3 rows are summed per node; a warp's 16 rows touch at most two adjacent
// nodes (16 < 20), so one masked butterfly per node suffices
template <int NNT>
__device__ __forceinline__ void rowsum_node(float (*acc)[4], int m0, int lane,
                                            float (*macc)[RRN_D], int nh) {
    int nodeA = m0 / RRN_E;
    bool two = (m0 + 15) / RRN_E != nodeA;
#pragma unroll
    for (int nt = 0; nt < NNT; ++nt) {
        int c = nh + nt * 8 + (lane & 3) * 2;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            int r = m0 + (lane >> 2) + h * 8;
            int node = r / RRN_E;
            float s0 = (node == nodeA) ? acc[nt][h * 2] : 0.f;
            float s1 = (node == nodeA) ? acc[nt][h * 2 + 1] : 0.f;
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, off);
                s1 += __shfl_xor_sync(0xffffffffu, s1, off);
            }
            if (lane < 4) {
                atomicAdd(&macc[nodeA][c], s0);
                atomicAdd(&macc[nodeA][c + 1], s1);
            }
            if (two) {
                s0 = (node == nodeA + 1) ? acc[nt][h * 2] : 0.f;
                s1 = (node == nodeA + 1) ? acc[nt][h * 2 + 1] : 0.f;
#pragma unroll
                for (int off = 4; off < 32; off <<= 1) {
                    s0 += __shfl_xor_sync(0xffffffffu, s0, off);
                    s1 += __shfl_xor_sync(0xffffffffu, s1, off);
                }
                if (lane < 4) {
                    atomicAdd(&macc[nodeA + 1][c], s0);
                    atomicAdd(&macc[nodeA + 1][c + 1], s1);
                }
            }
        }
    }
}

// e0 = relu(hw1[nb] + hw2[j] + b1) built straight into a smem tile;
// invalid tail nodes (j >= 81) read node 0 — finite garbage, never consumed
__device__ __forceinline__ void build_e0(bf16* tile, const bf16* hw12, const int* nb,
                                         const float* b1, int b, int j0, int tid) {
    for (int idx = tid; idx < RRN_ROWS * 12; idx += blockDim.x) {
        int r = idx / 12, cb = idx % 12;
        int l = r / RRN_E, k = r % RRN_E;
        int j = j0 + l;
        bool v = j < RRN_NN;
        int src = v ? nb[j * RRN_E + k] : 0;
        int jr = v ? j : 0;
        uint4 v1 = *(const uint4*)(hw12 + (size_t)(b * RRN_NN + src) * 192 + cb * 8);
        uint4 v2 = *(const uint4*)(hw12 + (size_t)(b * RRN_NN + jr) * 192 + 96 + cb * 8);
        uint4 vb0 = *(const uint4*)(b1 + cb * 8);
        uint4 vb1 = *(const uint4*)(b1 + cb * 8 + 4);
        const bf16* p1 = (const bf16*)&v1;
        const bf16* p2 = (const bf16*)&v2;
        const float* pb = (const float*)&vb0;
        const float* qb = (const float*)&vb1;
        uint4 o;
        bf16* po = (bf16*)&o;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            float bias = (i < 4) ? pb[i] : qb[i - 4];
            po[i] = __float2bfloat16(fmaxf(__bfloat162float(p1[i]) + __bfloat162float(p2[i]) + bias, 0.f));
        }
        *(uint4*)&tile[r * RRN_ASTR + cb * 8] = o;
    }
}
