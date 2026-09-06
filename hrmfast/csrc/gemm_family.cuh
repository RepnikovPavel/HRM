#pragma once
#include <cuda_bf16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

// tile configs proven on 2x RTX 5060 Ti (sm_120); sweep results in hrmfast/tests/sweep_gemm.py output
#define HRMFAST_CFGS \
    X(0, 64, 64, 64, 2, 128, 2, 2, 2) \
    X(1, 128, 64, 32, 2, 128, 2, 2, 2) \
    X(2, 128, 128, 64, 2, 256, 2, 4, 1) \
    X(3, 64, 128, 64, 2, 128, 2, 2, 1) \
    X(4, 64, 64, 32, 2, 128, 2, 2, 3)

#define HRMFAST_N_CFG 5

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return (uint32_t)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
    int sz = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(sz));
}

__device__ __forceinline__ void cp_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 64B rows (BK=32) need an 8-element pad to keep ldmatrix conflict-free;
// 128B rows (BK=64) use an XOR chunk swizzle instead, so no smem is wasted.
template <int ASTR>
__device__ __forceinline__ int swz(int c, int r) {
    return (ASTR % 64 == 0) ? (c ^ (r & 7)) : c;
}

// out[M,N] = sum_c a_c[M,K] @ b_c[N,K]^T, c < NB (b1 ignored when NB == 1).
// EPI 0: out = acc. EPI 1 (NB=2): out = silu(acc0) * acc1.
// EPI 2: out = bf16(resid + acc) and per-row sum-of-squares (of the stored bf16
// values, so backward sees exactly what was written) atomically added to sumsq[M].
// EPI 3 (NB=2): out = dg = aux*u*silu'(g), out2 = du = aux*silu(g), aux = dout;
// g,u stay in registers so the recomputed activations never touch global memory.
template <int BM, int BN, int BK, int STAGES, int THREADS, int WM, int WN, int MINB, int EPI, int NB>
__global__ void __launch_bounds__(THREADS, MINB) gemm_tn_kernel(
    const bf16* __restrict__ a,
    const bf16* __restrict__ b0,
    const bf16* __restrict__ b1,
    const bf16* __restrict__ aux,
    bf16* __restrict__ out,
    bf16* __restrict__ out2,
    float* __restrict__ sumsq,
    int M, int N, int K)
{
    constexpr int ASTR = (BK % 64 == 0) ? BK : BK + 8;
    constexpr int MT = BM / WM / 16;
    constexpr int NT = BN / WN / 8;
    constexpr int NACC = (EPI == 1 || EPI == 3) ? 2 : 1;
    constexpr int ASTAGE = BM * ASTR * 2;
    constexpr int BSTAGE = BN * ASTR * 2;
    constexpr int A_ITER = BM * BK / 8 / THREADS;
    constexpr int B_ITER = BN * BK / 8 / THREADS;

    extern __shared__ bf16 smem[];
    const uint32_t as_base = smem_u32(smem);
    const uint32_t b0_base = as_base + STAGES * ASTAGE;
    const uint32_t b1_base = b0_base + (NB - 1) * STAGES * BSTAGE;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = (warp / WN) * (BM / WM);
    const int warp_n = (warp % WN) * (BN / WN);

    const int ktiles = (K + BK - 1) / BK;

    float acc[NACC][MT][NT][4];
#pragma unroll
    for (int w = 0; w < NACC; ++w)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[w][i][j][v] = 0.f;

    auto load_stage = [&](int stage, int kt) {
        int k0 = kt * BK;
        uint32_t sa = as_base + stage * ASTAGE;
        uint32_t s0 = b0_base + stage * BSTAGE;
        uint32_t s1 = b1_base + stage * BSTAGE;
#pragma unroll
        for (int i = 0; i < A_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 8), c = idx % (BK / 8);
            int gr = block_row + r, gc = k0 + c * 8;
            bool p = (gr < M) && (gc + 8 <= K);
            const bf16* src = p ? a + (long)gr * K + gc : a;
            cp_async16(sa + (r * ASTR + swz<ASTR>(c, r) * 8) * 2, src, p);
        }
#pragma unroll
        for (int i = 0; i < B_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 8), c = idx % (BK / 8);
            int gr = block_col + r, gc = k0 + c * 8;
            bool p = (gr < N) && (gc + 8 <= K);
            uint32_t dst = (r * ASTR + swz<ASTR>(c, r) * 8) * 2;
            const bf16* src0 = p ? b0 + (long)gr * K + gc : b0;
            cp_async16(s0 + dst, src0, p);
            if (NB == 2) {
                const bf16* src1 = p ? b1 + (long)gr * K + gc : b1;
                cp_async16(s1 + dst, src1, p);
            }
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_stage(s, s);
        cp_commit();
    }

    uint32_t afrag[2][MT][4];
    uint32_t bfrag[NACC][2][NT][2];

    const uint32_t a_row = (warp_m + (lane & 15)) * ASTR * 2;
    const int bh = lane >> 3;
    const uint32_t b_row = (warp_n + (bh >> 1) * 8 + (lane & 7)) * ASTR * 2;

    auto load_frags = [&](int stage, int ks, int buf) {
        uint32_t ab = as_base + stage * ASTAGE + a_row + swz<ASTR>(ks * 2 + (lane >> 4), lane) * 16;
        uint32_t sb0 = b0_base + stage * BSTAGE + b_row + swz<ASTR>(ks * 2 + (bh & 1), lane) * 16;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            ldsm_x4(ab + mt * 16 * ASTR * 2, afrag[buf][mt][0], afrag[buf][mt][1],
                    afrag[buf][mt][2], afrag[buf][mt][3]);
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) {
            ldsm_x4(sb0 + p * 16 * ASTR * 2, bfrag[0][buf][p * 2][0], bfrag[0][buf][p * 2][1],
                    bfrag[0][buf][p * 2 + 1][0], bfrag[0][buf][p * 2 + 1][1]);
        }
        if (NACC == 2) {
            uint32_t sb1 = b1_base + stage * BSTAGE + b_row + swz<ASTR>(ks * 2 + (bh & 1), lane) * 16;
#pragma unroll
            for (int p = 0; p < NT / 2; ++p) {
                ldsm_x4(sb1 + p * 16 * ASTR * 2, bfrag[1][buf][p * 2][0], bfrag[1][buf][p * 2][1],
                        bfrag[1][buf][p * 2 + 1][0], bfrag[1][buf][p * 2 + 1][1]);
            }
        }
    };

    auto mma_tile = [&](int buf) {
#pragma unroll
        for (int w = 0; w < NACC; ++w)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    mma_bf16(acc[w][mt][nt][0], acc[w][mt][nt][1], acc[w][mt][nt][2], acc[w][mt][nt][3],
                             afrag[buf][mt][0], afrag[buf][mt][1], afrag[buf][mt][2], afrag[buf][mt][3],
                             bfrag[w][buf][nt][0], bfrag[w][buf][nt][1]);
    };

    cp_wait<STAGES - 2>();
    __syncthreads();
    load_frags(0, 0, 0);

    int read = 0, write = STAGES - 1;
    for (int kt = 0; kt < ktiles; ++kt) {
        int nread = (read + 1) % STAGES;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            if (ks == BK / 16 - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                load_frags(nread, 0, 0);
            } else {
                load_frags(read, ks + 1, (ks + 1) & 1);
            }
            if (ks == 0) {
                if (kt + STAGES - 1 < ktiles) load_stage(write, kt + STAGES - 1);
                cp_commit();
            }
            mma_tile(ks & 1);
        }
        read = nread;
        write = (write + 1) % STAGES;
    }

    __shared__ float rs[EPI == 2 ? BM : 1];
    if (EPI == 2) {
        for (int i = tid; i < BM; i += THREADS) rs[i] = 0.f;
        __syncthreads();
    }

    float rsum[EPI == 2 ? MT : 1][EPI == 2 ? 2 : 1];
    if (EPI == 2)
#pragma unroll
        for (int i = 0; i < MT; ++i) rsum[i][0] = rsum[i][1] = 0.f;

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        int r0 = block_row + warp_m + mt * 16 + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int c0 = block_col + warp_n + nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = r0 + h * 8;
                if (r >= M) continue;
                float v0 = acc[0][mt][nt][h * 2];
                float v1 = acc[0][mt][nt][h * 2 + 1];
                float w0 = 0.f, w1 = 0.f;
                if (EPI == 1) {
                    float g0 = v0, g1 = v1;
                    float u0 = acc[1][mt][nt][h * 2], u1 = acc[1][mt][nt][h * 2 + 1];
                    v0 = g0 * __fdividef(1.f, 1.f + __expf(-g0)) * u0;
                    v1 = g1 * __fdividef(1.f, 1.f + __expf(-g1)) * u1;
                } else if (EPI == 2) {
                    float rv0 = 0.f, rv1 = 0.f;
                    if (c0 + 1 < N) {
                        __nv_bfloat162 rv = *(__nv_bfloat162*)(aux + (long)r * N + c0);
                        rv0 = __bfloat162float(rv.x);
                        rv1 = __bfloat162float(rv.y);
                    } else if (c0 < N) {
                        rv0 = __bfloat162float(aux[(long)r * N + c0]);
                    }
                    v0 += rv0;
                    v1 += rv1;
                } else if (EPI == 3) {
                    float g0 = v0, g1 = v1;
                    float u0 = acc[1][mt][nt][h * 2], u1 = acc[1][mt][nt][h * 2 + 1];
                    float s0 = __fdividef(1.f, 1.f + __expf(-g0));
                    float s1 = __fdividef(1.f, 1.f + __expf(-g1));
                    float d0 = 0.f, d1 = 0.f;
                    if (c0 + 1 < N) {
                        __nv_bfloat162 dp = *(__nv_bfloat162*)(aux + (long)r * N + c0);
                        d0 = __bfloat162float(dp.x);
                        d1 = __bfloat162float(dp.y);
                    } else if (c0 < N) {
                        d0 = __bfloat162float(aux[(long)r * N + c0]);
                    }
                    v0 = d0 * u0 * s0 * (1.f + g0 * (1.f - s0));
                    v1 = d1 * u1 * s1 * (1.f + g1 * (1.f - s1));
                    w0 = d0 * g0 * s0;
                    w1 = d1 * g1 * s1;
                }
                if (c0 + 1 < N) {
                    __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(v0, v1));
                    *(__nv_bfloat162*)(out + (long)r * N + c0) = pk;
                    if (EPI == 3) {
                        __nv_bfloat162 pk2 = __float22bfloat162_rn(make_float2(w0, w1));
                        *(__nv_bfloat162*)(out2 + (long)r * N + c0) = pk2;
                    }
                    if (EPI == 2) {
                        rsum[mt][h] += __bfloat162float(pk.x) * __bfloat162float(pk.x);
                        rsum[mt][h] += __bfloat162float(pk.y) * __bfloat162float(pk.y);
                    }
                } else if (c0 < N) {
                    bf16 sv = __float2bfloat16(v0);
                    out[(long)r * N + c0] = sv;
                    if (EPI == 3) {
                        out2[(long)r * N + c0] = __float2bfloat16(w0);
                    }
                    if (EPI == 2) {
                        float f = __bfloat162float(sv);
                        rsum[mt][h] += f * f;
                    }
                }
            }
        }
    }

    if (EPI == 2) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int lr = warp_m + mt * 16 + (lane >> 2) + h * 8;
                atomicAdd(&rs[lr], rsum[mt][h]);
            }
        __syncthreads();
        for (int i = tid; i < BM; i += THREADS) {
            int r = block_row + i;
            if (r < M) atomicAdd(&sumsq[r], rs[i]);
        }
    }
}

// out[M,K] = sum_c a_c[M,N] @ b_c[N,K], c < NB. B is [N,K] row-major, so its
// fragments come from ldmatrix.trans (k rows are the reduction dim).
template <int BM, int BN, int BK, int STAGES, int THREADS, int WM, int WN, int MINB, int NB>
__global__ void __launch_bounds__(THREADS, MINB) gemm_nn_kernel(
    const bf16* __restrict__ a0,
    const bf16* __restrict__ a1,
    const bf16* __restrict__ b0,
    const bf16* __restrict__ b1,
    bf16* __restrict__ out,
    int M, int N, int K)
{
    constexpr int ASTR = (BK % 64 == 0) ? BK : BK + 8;
    constexpr int BSTR = (BN % 64 == 0) ? BN : BN + 8;
    constexpr int MT = BM / WM / 16;
    constexpr int NT = BN / WN / 8;
    constexpr int ASTAGE = BM * ASTR * 2;
    constexpr int BSTAGE = BK * BSTR * 2;
    constexpr int A_ITER = BM * BK / 8 / THREADS;
    constexpr int B_ITER = BK * BN / 8 / THREADS;

    extern __shared__ bf16 smem[];
    const uint32_t as_base = smem_u32(smem);
    const uint32_t bs_base = as_base + STAGES * ASTAGE;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = (warp / WN) * (BM / WM);
    const int warp_n = (warp % WN) * (BN / WN);

    const int ktp = (N + BK - 1) / BK;
    const int ktiles = NB * ktp;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.f;

    auto load_stage = [&](int stage, int kt) {
        int half = (kt < ktp) ? 0 : 1;
        int k0 = (kt - half * ktp) * BK;
        const bf16* ga = half ? a1 : a0;
        const bf16* gb = half ? b1 : b0;
        uint32_t sa = as_base + stage * ASTAGE;
        uint32_t sb = bs_base + stage * BSTAGE;
#pragma unroll
        for (int i = 0; i < A_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BK / 8), c = idx % (BK / 8);
            int gr = block_row + r, gc = k0 + c * 8;
            bool p = (gr < M) && (gc + 8 <= N);
            const bf16* src = p ? ga + (long)gr * N + gc : ga;
            cp_async16(sa + (r * ASTR + swz<ASTR>(c, r) * 8) * 2, src, p);
        }
#pragma unroll
        for (int i = 0; i < B_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BN / 8), c = idx % (BN / 8);
            int gr = k0 + r, gc = block_col + c * 8;
            bool p = (gr < N) && (gc + 8 <= K);
            const bf16* src = p ? gb + (long)gr * K + gc : gb;
            cp_async16(sb + (r * BSTR + swz<BSTR>(c, r) * 8) * 2, src, p);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_stage(s, s);
        cp_commit();
    }

    uint32_t afrag[2][MT][4];
    uint32_t bfrag[2][NT][2];

    const uint32_t a_row = (warp_m + (lane & 15)) * ASTR * 2;
    const int bh = lane >> 3;
    const uint32_t b_row = ((bh & 1) * 8 + (lane & 7)) * BSTR * 2;

    auto load_frags = [&](int stage, int ks, int buf) {
        uint32_t ab = as_base + stage * ASTAGE + a_row + swz<ASTR>(ks * 2 + (lane >> 4), lane) * 16;
        uint32_t bb = bs_base + stage * BSTAGE + b_row + ks * 16 * BSTR * 2;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            ldsm_x4(ab + mt * 16 * ASTR * 2, afrag[buf][mt][0], afrag[buf][mt][1],
                    afrag[buf][mt][2], afrag[buf][mt][3]);
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) {
            uint32_t addr = bb + swz<BSTR>(warp_n / 8 + p * 2 + (bh >> 1), lane) * 16;
            ldsm_x4_trans(addr, bfrag[buf][p * 2][0], bfrag[buf][p * 2][1],
                          bfrag[buf][p * 2 + 1][0], bfrag[buf][p * 2 + 1][1]);
        }
    };

    auto mma_tile = [&](int buf) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
                mma_bf16(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                         afrag[buf][mt][0], afrag[buf][mt][1], afrag[buf][mt][2], afrag[buf][mt][3],
                         bfrag[buf][nt][0], bfrag[buf][nt][1]);
    };

    cp_wait<STAGES - 2>();
    __syncthreads();
    load_frags(0, 0, 0);

    int read = 0, write = STAGES - 1;
    for (int kt = 0; kt < ktiles; ++kt) {
        int nread = (read + 1) % STAGES;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            if (ks == BK / 16 - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                load_frags(nread, 0, 0);
            } else {
                load_frags(read, ks + 1, (ks + 1) & 1);
            }
            if (ks == 0) {
                if (kt + STAGES - 1 < ktiles) load_stage(write, kt + STAGES - 1);
                cp_commit();
            }
            mma_tile(ks & 1);
        }
        read = nread;
        write = (write + 1) % STAGES;
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        int r0 = block_row + warp_m + mt * 16 + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            int c0 = block_col + warp_n + nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int r = r0 + h * 8;
                if (r >= M) continue;
                if (c0 + 1 < K) {
                    __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(acc[mt][nt][h * 2], acc[mt][nt][h * 2 + 1]));
                    *(__nv_bfloat162*)(out + (long)r * K + c0) = pk;
                } else if (c0 < K) {
                    out[(long)r * K + c0] = __float2bfloat16(acc[mt][nt][h * 2]);
                }
            }
        }
    }
}

// ws_c[N,K] (+)= a_c[M,N]^T @ b[M,K], c < NB; M reduction split over gridDim.z
// slices, fp32 atomic partials (order varies; error far below bf16 rounding).
template <int BM, int BN, int BK, int STAGES, int THREADS, int WM, int WN, int MINB, int NB>
__global__ void __launch_bounds__(THREADS, MINB) gemm_nt_kernel(
    const bf16* __restrict__ a0,
    const bf16* __restrict__ a1,
    const bf16* __restrict__ b,
    float* __restrict__ ws0,
    float* __restrict__ ws1,
    int M, int N, int K)
{
    constexpr int ASTR = (BM % 64 == 0) ? BM : BM + 8;
    constexpr int BSTR = (BN % 64 == 0) ? BN : BN + 8;
    constexpr int MT = BM / WM / 16;
    constexpr int NT = BN / WN / 8;
    constexpr int ASTAGE = BK * ASTR * 2;
    constexpr int BSTAGE = BK * BSTR * 2;
    constexpr int A_ITER = BK * BM / 8 / THREADS;
    constexpr int B_ITER = BK * BN / 8 / THREADS;

    extern __shared__ bf16 smem[];
    const uint32_t a0_base = smem_u32(smem);
    const uint32_t a1_base = a0_base + (NB - 1) * STAGES * ASTAGE;
    const uint32_t bx_base = a0_base + NB * STAGES * ASTAGE;

    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int warp_m = (warp / WN) * (BM / WM);
    const int warp_n = (warp % WN) * (BN / WN);

    const int ktiles_all = (M + BK - 1) / BK;
    const int per_slice = (ktiles_all + (int)gridDim.z - 1) / (int)gridDim.z;
    const int kt0 = blockIdx.z * per_slice;
    const int ktiles = min(ktiles_all, kt0 + per_slice) - kt0;
    if (ktiles <= 0) return;

    float acc[NB][MT][NT][4];
#pragma unroll
    for (int w = 0; w < NB; ++w)
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int v = 0; v < 4; ++v) acc[w][i][j][v] = 0.f;

    auto load_stage = [&](int stage, int kt) {
        int m0 = (kt0 + kt) * BK;
        uint32_t s0 = a0_base + stage * ASTAGE;
        uint32_t s1 = a1_base + stage * ASTAGE;
        uint32_t sx = bx_base + stage * BSTAGE;
#pragma unroll
        for (int i = 0; i < A_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BM / 8), c = idx % (BM / 8);
            int gr = m0 + r, gc = block_row + c * 8;
            bool p = (gr < M) && (gc + 8 <= N);
            uint32_t dst = (r * ASTR + swz<ASTR>(c, r) * 8) * 2;
            const bf16* src0 = p ? a0 + (long)gr * N + gc : a0;
            cp_async16(s0 + dst, src0, p);
            if (NB == 2) {
                const bf16* src1 = p ? a1 + (long)gr * N + gc : a1;
                cp_async16(s1 + dst, src1, p);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITER; ++i) {
            int idx = tid + i * THREADS;
            int r = idx / (BN / 8), c = idx % (BN / 8);
            int gr = m0 + r, gc = block_col + c * 8;
            bool p = (gr < M) && (gc + 8 <= K);
            const bf16* src = p ? b + (long)gr * K + gc : b;
            cp_async16(sx + (r * BSTR + swz<BSTR>(c, r) * 8) * 2, src, p);
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_stage(s, s);
        cp_commit();
    }

    uint32_t afrag[NB][2][MT][4];
    uint32_t bfrag[2][NT][2];

    const int bh = lane >> 3;
    const uint32_t a_row = ((bh >> 1) * 8 + (lane & 7)) * ASTR * 2;
    const uint32_t b_row = ((bh & 1) * 8 + (lane & 7)) * BSTR * 2;

    auto load_frags = [&](int stage, int ks, int buf) {
        uint32_t sa0 = a0_base + stage * ASTAGE + a_row + ks * 16 * ASTR * 2;
        uint32_t bx = bx_base + stage * BSTAGE + b_row + ks * 16 * BSTR * 2;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            uint32_t off = swz<ASTR>(warp_m / 8 + mt * 2 + (bh & 1), lane) * 16;
            ldsm_x4_trans(sa0 + off, afrag[0][buf][mt][0], afrag[0][buf][mt][1],
                          afrag[0][buf][mt][2], afrag[0][buf][mt][3]);
            if (NB == 2) {
                uint32_t sa1 = a1_base + stage * ASTAGE + a_row + ks * 16 * ASTR * 2;
                ldsm_x4_trans(sa1 + off, afrag[1][buf][mt][0], afrag[1][buf][mt][1],
                              afrag[1][buf][mt][2], afrag[1][buf][mt][3]);
            }
        }
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) {
            uint32_t addr = bx + swz<BSTR>(warp_n / 8 + p * 2 + (bh >> 1), lane) * 16;
            ldsm_x4_trans(addr, bfrag[buf][p * 2][0], bfrag[buf][p * 2][1],
                          bfrag[buf][p * 2 + 1][0], bfrag[buf][p * 2 + 1][1]);
        }
    };

    auto mma_tile = [&](int buf) {
#pragma unroll
        for (int w = 0; w < NB; ++w)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                    mma_bf16(acc[w][mt][nt][0], acc[w][mt][nt][1], acc[w][mt][nt][2], acc[w][mt][nt][3],
                             afrag[w][buf][mt][0], afrag[w][buf][mt][1], afrag[w][buf][mt][2], afrag[w][buf][mt][3],
                             bfrag[buf][nt][0], bfrag[buf][nt][1]);
    };

    cp_wait<STAGES - 2>();
    __syncthreads();
    load_frags(0, 0, 0);

    int read = 0, write = STAGES - 1;
    for (int kt = 0; kt < ktiles; ++kt) {
        int nread = (read + 1) % STAGES;
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
            if (ks == BK / 16 - 1) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                load_frags(nread, 0, 0);
            } else {
                load_frags(read, ks + 1, (ks + 1) & 1);
            }
            if (ks == 0) {
                if (kt + STAGES - 1 < ktiles) load_stage(write, kt + STAGES - 1);
                cp_commit();
            }
            mma_tile(ks & 1);
        }
        read = nread;
        write = (write + 1) % STAGES;
    }

#pragma unroll
    for (int w = 0; w < NB; ++w) {
        float* ws = w ? ws1 : ws0;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            int r0 = block_row + warp_m + mt * 16 + (lane >> 2);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int c0 = block_col + warp_n + nt * 8 + (lane & 3) * 2;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    int r = r0 + h * 8;
                    if (r >= N) continue;
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        int c = c0 + v;
                        if (c >= K) continue;
                        atomicAdd(ws + (long)r * K + c, acc[w][mt][nt][h * 2 + v]);
                    }
                }
            }
        }
    }
}

template <int EPI, int NB>
void tn_launch(int cfg, const bf16* a, const bf16* b0, const bf16* b1, const bf16* aux,
               bf16* out, bf16* out2, float* sumsq, int M, int N, int K, cudaStream_t stream) {
    switch (cfg) {
#define X(id, bm, bn, bk, st, th, wm, wn, mb) \
        case id: { \
            constexpr int ASTR = (bk % 64 == 0) ? bk : bk + 8; \
            const int smem = st * (bm + NB * bn) * ASTR * 2; \
            static bool cfgd = false; \
            if (!cfgd) { \
                cudaFuncSetAttribute(gemm_tn_kernel<bm, bn, bk, st, th, wm, wn, mb, EPI, NB>, \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                cfgd = true; \
            } \
            dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm); \
            gemm_tn_kernel<bm, bn, bk, st, th, wm, wn, mb, EPI, NB><<<grid, th, smem, stream>>>( \
                a, b0, b1, aux, out, out2, sumsq, M, N, K); \
            break; }
        HRMFAST_CFGS
#undef X
        default: break;
    }
}

template <int NB>
void nn_launch(int cfg, const bf16* a0, const bf16* a1, const bf16* b0, const bf16* b1,
               bf16* out, int M, int N, int K, cudaStream_t stream) {
    switch (cfg) {
#define X(id, bm, bn, bk, st, th, wm, wn, mb) \
        case id: { \
            constexpr int ASTR = (bk % 64 == 0) ? bk : bk + 8; \
            constexpr int SB = (bn % 64 == 0) ? bn : bn + 8; \
            const int smem = st * (bm * ASTR + bk * SB) * 2; \
            static bool cfgd = false; \
            if (!cfgd) { \
                cudaFuncSetAttribute(gemm_nn_kernel<bm, bn, bk, st, th, wm, wn, mb, NB>, \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                cfgd = true; \
            } \
            dim3 grid((K + bn - 1) / bn, (M + bm - 1) / bm); \
            gemm_nn_kernel<bm, bn, bk, st, th, wm, wn, mb, NB><<<grid, th, smem, stream>>>( \
                a0, a1, b0, b1, out, M, N, K); \
            break; }
        HRMFAST_CFGS
#undef X
        default: break;
    }
}

#define HRMFAST_NT_SPLIT 8

template <int NB>
void nt_launch(int cfg, const bf16* a0, const bf16* a1, const bf16* b,
               float* ws0, float* ws1, int M, int N, int K, cudaStream_t stream) {
    switch (cfg) {
#define X(id, bm, bn, bk, st, th, wm, wn, mb) \
        case id: { \
            constexpr int SA = (bm % 64 == 0) ? bm : bm + 8; \
            constexpr int SB = (bn % 64 == 0) ? bn : bn + 8; \
            const int smem = st * bk * (NB * SA + SB) * 2; \
            static bool cfgd = false; \
            if (!cfgd) { \
                cudaFuncSetAttribute(gemm_nt_kernel<bm, bn, bk, st, th, wm, wn, mb, NB>, \
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                cfgd = true; \
            } \
            dim3 grid((K + bn - 1) / bn, (N + bm - 1) / bm, HRMFAST_NT_SPLIT); \
            gemm_nt_kernel<bm, bn, bk, st, th, wm, wn, mb, NB><<<grid, th, smem, stream>>>( \
                a0, a1, b, ws0, ws1, M, N, K); \
            break; }
        HRMFAST_CFGS
#undef X
        default: break;
    }
}
