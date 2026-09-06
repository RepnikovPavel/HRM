#pragma once
#include "gemm_family.cuh"

// Fused attention core for [B, S, H, 64] bf16, causal=False, S <= 128.
// The whole score row fits in registers (S <= 128), so softmax needs no online
// rescaling: one pass over K/V per q-tile. The mma accumulator layout of the
// scores matches the A-operand fragment layout of the P@V / dS@K mmas, so P
// and dS are repacked in registers and never touch smem. Forward stores lse
// for backward. Backward: B1 recomputes P per q-tile and produces dq +
// rowsum(dO*O); B2 is s-tiled so each block owns its dK/dV rows (no atomics).

#define ATTN_D 64

// non-trans A fragment (m16k16) from smem tile with row stride STR elements
template <int STR>
__device__ __forceinline__ void ldsm_a(uint32_t base, int r0, int ks, int lane, uint32_t* a) {
    uint32_t addr = base + (r0 + (lane & 15)) * STR * 2 + swz<STR>(ks * 2 + (lane >> 4), lane) * 16;
    ldsm_x4(addr, a[0], a[1], a[2], a[3]);
}

// non-trans B pair (k16 n16, two n8 frags) from smem [n][k] tile (weights layout)
template <int STR>
__device__ __forceinline__ void ldsm_b(uint32_t base, int n0, int ks, int lane, uint32_t* b0, uint32_t* b1) {
    int bh = lane >> 3;
    uint32_t addr = base + (n0 + (bh >> 1) * 8 + (lane & 7)) * STR * 2 + swz<STR>(ks * 2 + (bh & 1), lane) * 16;
    ldsm_x4(addr, b0[0], b0[1], b1[0], b1[1]);
}

// trans B pair (k16 n16) from smem [k][n] tile (activations layout)
template <int STR>
__device__ __forceinline__ void ldsm_bt(uint32_t base, int n0, int ks, int lane, uint32_t* b0, uint32_t* b1) {
    int bh = lane >> 3;
    uint32_t addr = base + (ks * 16 + (bh & 1) * 8 + (lane & 7)) * STR * 2 + swz<STR>(n0 / 8 + (bh >> 1), lane) * 16;
    ldsm_x4_trans(addr, b0[0], b0[1], b1[0], b1[1]);
}

// trans A fragment (m16k16) from smem [k][m] tile
template <int STR>
__device__ __forceinline__ void ldsm_at(uint32_t base, int m0, int ks, int lane, uint32_t* a) {
    int bh = lane >> 3;
    uint32_t addr = base + (ks * 16 + (bh >> 1) * 8 + (lane & 7)) * STR * 2 + swz<STR>(m0 / 8 + (bh & 1), lane) * 16;
    ldsm_x4_trans(addr, a[0], a[1], a[2], a[3]);
}

__device__ __forceinline__ uint32_t pack_bf16(float x, float y) {
    __nv_bfloat162 p = __float22bfloat162_rn(make_float2(x, y));
    return *(uint32_t*)&p;
}

template <int SP>
__global__ void __launch_bounds__(128, 2) attn_fwd_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, float* __restrict__ lse,
    int S, int H, float scale, long sb_q, long sb_k, long sb_v, long ss_q, long ss_k, long ss_v)
{
    extern __shared__ bf16 smem[];
    bf16* Qs = smem;
    bf16* Ks = Qs + SP * ATTN_D;
    bf16* Vs = Ks + SP * ATTN_D;

    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const long hb = (long)H * ATTN_D;
    const bf16* qb = q + (long)b * sb_q + h * ATTN_D;
    const bf16* kb = k + (long)b * sb_k + h * ATTN_D;
    const bf16* vb = v + (long)b * sb_v + h * ATTN_D;

    auto load_mat = [&](const bf16* src, long ss, bf16* dst) {
        for (int idx = tid; idx < SP * (ATTN_D / 8); idx += 128) {
            int r = idx >> 3, c = idx & 7;
            bool p = r < S;
            cp_async16(smem_u32(dst + r * ATTN_D + swz<ATTN_D>(c, r) * 8),
                       p ? src + (long)r * ss + c * 8 : src, p);
        }
    };
    load_mat(qb, ss_q, Qs);
    load_mat(kb, ss_k, Ks);
    load_mat(vb, ss_v, Vs);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const uint32_t qbase = smem_u32(Qs);
    const uint32_t kbase = smem_u32(Ks);
    const uint32_t vbase = smem_u32(Vs);
    constexpr int NNT = SP / 8;
    bf16* ob = o + ((long)b * S) * hb + h * ATTN_D;

    for (int t = warp; t < SP / 16; t += 4) {
        float sco[NNT][4];
#pragma unroll
        for (int nt = 0; nt < NNT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) sco[nt][i] = 0.f;

#pragma unroll
        for (int ks = 0; ks < ATTN_D / 16; ++ks) {
            uint32_t a[4];
            ldsm_a<ATTN_D>(qbase, t * 16, ks, lane, a);
#pragma unroll
            for (int ntp = 0; ntp < NNT / 2; ++ntp) {
                uint32_t bg[2], bu[2];
                ldsm_b<ATTN_D>(kbase, ntp * 16, ks, lane, bg, bu);
                mma_bf16(sco[ntp * 2][0], sco[ntp * 2][1], sco[ntp * 2][2], sco[ntp * 2][3],
                         a[0], a[1], a[2], a[3], bg[0], bg[1]);
                mma_bf16(sco[ntp * 2 + 1][0], sco[ntp * 2 + 1][1], sco[ntp * 2 + 1][2], sco[ntp * 2 + 1][3],
                         a[0], a[1], a[2], a[3], bu[0], bu[1]);
            }
        }

        const int r0 = t * 16 + (lane >> 2);
        float mx[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int nt = 0; nt < NNT; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int ci = c + (i & 1);
                float s = (ci < S) ? sco[nt][i] * scale : -1e30f;
                mx[i >> 1] = fmaxf(mx[i >> 1], s);
                sco[nt][i] = s;
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mx[0] = fmaxf(mx[0], __shfl_xor_sync(0xffffffffu, mx[0], off));
            mx[1] = fmaxf(mx[1], __shfl_xor_sync(0xffffffffu, mx[1], off));
        }
        float sum[2] = {0.f, 0.f};
#pragma unroll
        for (int nt = 0; nt < NNT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float p = __expf(sco[nt][i] - mx[i >> 1]);
                sco[nt][i] = p;
                sum[i >> 1] += p;
            }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            sum[0] += __shfl_xor_sync(0xffffffffu, sum[0], off);
            sum[1] += __shfl_xor_sync(0xffffffffu, sum[1], off);
        }
        float rsum[2] = {__fdividef(1.f, fmaxf(sum[0], 1e-30f)), __fdividef(1.f, fmaxf(sum[1], 1e-30f))};
        if ((lane & 3) == 0) {
            if (r0 < S) lse[(long)bh * S + r0] = mx[0] + __logf(sum[0]);
            if (r0 + 8 < S) lse[(long)bh * S + r0 + 8] = mx[1] + __logf(sum[1]);
        }

        float acco[ATTN_D / 8][4];
#pragma unroll
        for (int nt = 0; nt < ATTN_D / 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acco[nt][i] = 0.f;

#pragma unroll
        for (int ks = 0; ks < SP / 16; ++ks) {
            uint32_t a[4];
            a[0] = pack_bf16(sco[2 * ks][0] * rsum[0], sco[2 * ks][1] * rsum[0]);
            a[1] = pack_bf16(sco[2 * ks][2] * rsum[1], sco[2 * ks][3] * rsum[1]);
            a[2] = pack_bf16(sco[2 * ks + 1][0] * rsum[0], sco[2 * ks + 1][1] * rsum[0]);
            a[3] = pack_bf16(sco[2 * ks + 1][2] * rsum[1], sco[2 * ks + 1][3] * rsum[1]);
#pragma unroll
            for (int ntp = 0; ntp < ATTN_D / 16; ++ntp) {
                uint32_t bg[2], bu[2];
                ldsm_bt<ATTN_D>(vbase, ntp * 16, ks, lane, bg, bu);
                mma_bf16(acco[ntp * 2][0], acco[ntp * 2][1], acco[ntp * 2][2], acco[ntp * 2][3],
                         a[0], a[1], a[2], a[3], bg[0], bg[1]);
                mma_bf16(acco[ntp * 2 + 1][0], acco[ntp * 2 + 1][1], acco[ntp * 2 + 1][2], acco[ntp * 2 + 1][3],
                         a[0], a[1], a[2], a[3], bu[0], bu[1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < ATTN_D / 8; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                int r = r0 + hh * 8;
                if (r >= S) continue;
                __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(acco[nt][hh * 2], acco[nt][hh * 2 + 1]));
                *(__nv_bfloat162*)(ob + (long)r * hb + c) = pk;
            }
        }
    }
}

template <int SP>
__global__ void __launch_bounds__(128, 2) attn_bwd_dq_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const bf16* __restrict__ o, const bf16* __restrict__ dout, const float* __restrict__ lse,
    bf16* __restrict__ dq, float* __restrict__ dsum,
    int S, int H, float scale, long sb_q, long sb_k, long sb_v, long sb_o, long sb_do,
    long ss_q, long ss_k, long ss_v, long ss_o, long ss_do)
{
    extern __shared__ bf16 smem[];
    bf16* Qs = smem;
    bf16* Ks = Qs + SP * ATTN_D;
    bf16* Vs = Ks + SP * ATTN_D;
    bf16* DOs = Vs + SP * ATTN_D;

    const int bh = blockIdx.x;
    const int b = bh / H, h = bh % H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const long hb = (long)H * ATTN_D;
    auto load_mat = [&](const bf16* src, long sb, long ss, bf16* dst) {
        const bf16* base = src + (long)b * sb + h * ATTN_D;
        for (int idx = tid; idx < SP * (ATTN_D / 8); idx += 128) {
            int r = idx >> 3, c = idx & 7;
            bool p = r < S;
            cp_async16(smem_u32(dst + r * ATTN_D + swz<ATTN_D>(c, r) * 8),
                       p ? base + (long)r * ss + c * 8 : base, p);
        }
    };
    load_mat(q, sb_q, ss_q, Qs);
    load_mat(k, sb_k, ss_k, Ks);
    load_mat(v, sb_v, ss_v, Vs);
    load_mat(dout, sb_do, ss_do, DOs);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const uint32_t qbase = smem_u32(Qs);
    const uint32_t kbase = smem_u32(Ks);
    const uint32_t vbase = smem_u32(Vs);
    const uint32_t dobase = smem_u32(DOs);
    const bf16* ob = o + (long)b * sb_o + h * ATTN_D;
    constexpr int NNT = SP / 8;
    bf16* dqob = dq + ((long)b * S) * hb + h * ATTN_D;

    for (int t = warp; t < SP / 16; t += 4) {
        float sco[NNT][4], dpr[NNT][4];
#pragma unroll
        for (int nt = 0; nt < NNT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                sco[nt][i] = 0.f;
                dpr[nt][i] = 0.f;
            }

#pragma unroll
        for (int ks = 0; ks < ATTN_D / 16; ++ks) {
            uint32_t aq[4], ad[4];
            ldsm_a<ATTN_D>(qbase, t * 16, ks, lane, aq);
            ldsm_a<ATTN_D>(dobase, t * 16, ks, lane, ad);
#pragma unroll
            for (int ntp = 0; ntp < NNT / 2; ++ntp) {
                uint32_t bg[2], bu[2], vg[2], vu[2];
                ldsm_b<ATTN_D>(kbase, ntp * 16, ks, lane, bg, bu);
                ldsm_b<ATTN_D>(vbase, ntp * 16, ks, lane, vg, vu);
                mma_bf16(sco[ntp * 2][0], sco[ntp * 2][1], sco[ntp * 2][2], sco[ntp * 2][3],
                         aq[0], aq[1], aq[2], aq[3], bg[0], bg[1]);
                mma_bf16(sco[ntp * 2 + 1][0], sco[ntp * 2 + 1][1], sco[ntp * 2 + 1][2], sco[ntp * 2 + 1][3],
                         aq[0], aq[1], aq[2], aq[3], bu[0], bu[1]);
                mma_bf16(dpr[ntp * 2][0], dpr[ntp * 2][1], dpr[ntp * 2][2], dpr[ntp * 2][3],
                         ad[0], ad[1], ad[2], ad[3], vg[0], vg[1]);
                mma_bf16(dpr[ntp * 2 + 1][0], dpr[ntp * 2 + 1][1], dpr[ntp * 2 + 1][2], dpr[ntp * 2 + 1][3],
                         ad[0], ad[1], ad[2], ad[3], vu[0], vu[1]);
            }
        }

        const int r0 = t * 16 + (lane >> 2);
        float le[2] = {0.f, 0.f};
        if (r0 < S) le[0] = lse[(long)bh * S + r0];
        if (r0 + 8 < S) le[1] = lse[(long)bh * S + r0 + 8];

#pragma unroll
        for (int nt = 0; nt < NNT; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int ci = c + (i & 1);
                int r = r0 + (i >> 1) * 8;
                sco[nt][i] = (ci < S && r < S) ? __expf(sco[nt][i] * scale - le[i >> 1]) : 0.f;
            }
        }

        // D_i = sum_d dO[i,d]*O[i,d]; dO from smem, O from global (L2-hot)
        float dsumv[2] = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int d = (lane & 3) * 2 + j * 8;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                int lr = t * 16 + (lane >> 2) + hh * 8;
                float ov0 = 0.f, ov1 = 0.f;
                if (lr < S) {
                    __nv_bfloat162 oo = *(__nv_bfloat162*)(ob + (long)lr * ss_o + d);
                    ov0 = __bfloat162float(oo.x);
                    ov1 = __bfloat162float(oo.y);
                }
                uint32_t off = (lr * ATTN_D + swz<ATTN_D>(d >> 3, lr) * 8 + (d & 7)) * 2;
                __nv_bfloat162 dd = *(__nv_bfloat162*)((char*)DOs + off);
                dsumv[hh] += __bfloat162float(dd.x) * ov0 + __bfloat162float(dd.y) * ov1;
            }
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            dsumv[0] += __shfl_xor_sync(0xffffffffu, dsumv[0], off);
            dsumv[1] += __shfl_xor_sync(0xffffffffu, dsumv[1], off);
        }
        if ((lane & 3) == 0) {
            if (r0 < S) dsum[(long)bh * S + r0] = dsumv[0];
            if (r0 + 8 < S) dsum[(long)bh * S + r0 + 8] = dsumv[1];
        }

        float acco[ATTN_D / 8][4];
#pragma unroll
        for (int nt = 0; nt < ATTN_D / 8; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i) acco[nt][i] = 0.f;

#pragma unroll
        for (int ks = 0; ks < SP / 16; ++ks) {
            uint32_t a[4];
            float d0 = dsumv[0], d1 = dsumv[1];
            a[0] = pack_bf16(sco[2 * ks][0] * (dpr[2 * ks][0] - d0), sco[2 * ks][1] * (dpr[2 * ks][1] - d0));
            a[1] = pack_bf16(sco[2 * ks][2] * (dpr[2 * ks][2] - d1), sco[2 * ks][3] * (dpr[2 * ks][3] - d1));
            a[2] = pack_bf16(sco[2 * ks + 1][0] * (dpr[2 * ks + 1][0] - d0), sco[2 * ks + 1][1] * (dpr[2 * ks + 1][1] - d0));
            a[3] = pack_bf16(sco[2 * ks + 1][2] * (dpr[2 * ks + 1][2] - d1), sco[2 * ks + 1][3] * (dpr[2 * ks + 1][3] - d1));
#pragma unroll
            for (int ntp = 0; ntp < ATTN_D / 16; ++ntp) {
                uint32_t bg[2], bu[2];
                ldsm_bt<ATTN_D>(kbase, ntp * 16, ks, lane, bg, bu);
                mma_bf16(acco[ntp * 2][0], acco[ntp * 2][1], acco[ntp * 2][2], acco[ntp * 2][3],
                         a[0], a[1], a[2], a[3], bg[0], bg[1]);
                mma_bf16(acco[ntp * 2 + 1][0], acco[ntp * 2 + 1][1], acco[ntp * 2 + 1][2], acco[ntp * 2 + 1][3],
                         a[0], a[1], a[2], a[3], bu[0], bu[1]);
            }
        }

#pragma unroll
        for (int nt = 0; nt < ATTN_D / 8; ++nt) {
            int c = nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                int r = r0 + hh * 8;
                if (r >= S) continue;
                __nv_bfloat162 pk = __float22bfloat162_rn(make_float2(acco[nt][hh * 2] * scale, acco[nt][hh * 2 + 1] * scale));
                *(__nv_bfloat162*)(dqob + (long)r * hb + c) = pk;
            }
        }
    }
}

template <int SP>
__global__ void __launch_bounds__(128, 2) attn_bwd_dkdv_kernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const bf16* __restrict__ dout, const float* __restrict__ lse, const float* __restrict__ dsum,
    bf16* __restrict__ dk, bf16* __restrict__ dv,
    int S, int H, float scale, long sb_q, long sb_k, long sb_v, long sb_do,
    long ss_q, long ss_k, long ss_v, long ss_do)
{
    constexpr int PSTR = 40;
    extern __shared__ bf16 smem[];
    bf16* Kt = smem;
    bf16* Vt = Kt + 32 * ATTN_D;
    bf16* Qt = Vt + 32 * ATTN_D;
    bf16* DOt = Qt + 3 * 16 * ATTN_D;
    bf16* Ps = DOt + 3 * 16 * ATTN_D;
    bf16* Ds = Ps + 16 * PSTR;

    const int bh = blockIdx.y;
    const int b = bh / H, h = bh % H;
    const int s0 = blockIdx.x * 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

    const long hb = (long)H * ATTN_D;
    const bf16* qb = q + (long)b * sb_q + h * ATTN_D;
    const bf16* kb = k + (long)b * sb_k + h * ATTN_D;
    const bf16* vb = v + (long)b * sb_v + h * ATTN_D;
    const bf16* dob = dout + (long)b * sb_do + h * ATTN_D;

    auto load_tile = [&](const bf16* src, long ss, bf16* dst, int r0, int rows) {
        for (int idx = tid; idx < rows * (ATTN_D / 8); idx += 128) {
            int r = idx >> 3, c = idx & 7;
            bool p = (r0 + r) < S;
            cp_async16(smem_u32(dst + r * ATTN_D + swz<ATTN_D>(c, r) * 8),
                       p ? src + (long)(r0 + r) * ss + c * 8 : src, p);
        }
    };
    load_tile(kb, ss_k, Kt, s0, 32);
    load_tile(vb, ss_v, Vt, s0, 32);
    cp_commit();

    const uint32_t ktbase = smem_u32(Kt);
    const uint32_t vtbase = smem_u32(Vt);

    float accdk[2][2][4];
    float accdv[2][2][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                accdk[i][j][w] = 0.f;
                accdv[i][j][w] = 0.f;
            }

    const int nq = (S + 15) / 16;
    load_tile(qb, ss_q, Qt, 0, 16);
    load_tile(dob, ss_do, DOt, 0, 16);
    cp_commit();

    // 3-buffer rotation: load(t+1) targets the buffer last read in iteration t-2,
    // which every warp has finished before this warp passed iteration t-1's barriers
    for (int t = 0; t < nq; ++t) {
        bf16* qt = Qt + (t % 3) * 16 * ATTN_D;
        bf16* dot = DOt + (t % 3) * 16 * ATTN_D;
        if (t + 1 < nq) {
            bf16* qn = Qt + ((t + 1) % 3) * 16 * ATTN_D;
            bf16* don = DOt + ((t + 1) % 3) * 16 * ATTN_D;
            load_tile(qb, ss_q, qn, (t + 1) * 16, 16);
            load_tile(dob, ss_do, don, (t + 1) * 16, 16);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        const uint32_t qtb = smem_u32(qt);
        const uint32_t dotb = smem_u32(dot);
        float sco[4] = {0.f, 0.f, 0.f, 0.f};
        float dpr[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < ATTN_D / 16; ++ks) {
            uint32_t aq[4], ad[4], bk[2], bv[2];
            ldsm_a<ATTN_D>(qtb, 0, ks, lane, aq);
            ldsm_a<ATTN_D>(dotb, 0, ks, lane, ad);
            ldsm_b<ATTN_D>(ktbase, warp * 8, ks, lane, bk, bv);
            mma_bf16(sco[0], sco[1], sco[2], sco[3], aq[0], aq[1], aq[2], aq[3], bk[0], bk[1]);
            ldsm_b<ATTN_D>(vtbase, warp * 8, ks, lane, bk, bv);
            mma_bf16(dpr[0], dpr[1], dpr[2], dpr[3], ad[0], ad[1], ad[2], ad[3], bk[0], bk[1]);
        }

        const int r0 = t * 16 + (lane >> 2);
        const int c = s0 + warp * 8 + (lane & 3) * 2;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            int r = r0 + hh * 8;
            float le = (r < S) ? lse[(long)bh * S + r] : 0.f;
            float du = (r < S) ? dsum[(long)bh * S + r] : 0.f;
            float pv[2], dsv[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int ci = c + i;
                pv[i] = (ci < S && r < S) ? __expf(sco[hh * 2 + i] * scale - le) : 0.f;
                dsv[i] = pv[i] * (dpr[hh * 2 + i] - du);
            }
            *(__nv_bfloat162*)(Ps + (hh * 8 + (lane >> 2)) * PSTR + warp * 8 + (lane & 3) * 2) =
                __float22bfloat162_rn(make_float2(pv[0], pv[1]));
            *(__nv_bfloat162*)(Ds + (hh * 8 + (lane >> 2)) * PSTR + warp * 8 + (lane & 3) * 2) =
                __float22bfloat162_rn(make_float2(dsv[0], dsv[1]));
        }
        __syncthreads();

        uint32_t bq[2], bd[2], bq2[2], bd2[2];
        ldsm_bt<ATTN_D>(dotb, warp * 16, 0, lane, bd, bd2);
        ldsm_bt<ATTN_D>(qtb, warp * 16, 0, lane, bq, bq2);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            uint32_t amv[4], ams[4];
            ldsm_at<PSTR>(smem_u32(Ps), mt * 16, 0, lane, amv);
            ldsm_at<PSTR>(smem_u32(Ds), mt * 16, 0, lane, ams);
            mma_bf16(accdv[mt][0][0], accdv[mt][0][1], accdv[mt][0][2], accdv[mt][0][3],
                     amv[0], amv[1], amv[2], amv[3], bd[0], bd[1]);
            mma_bf16(accdv[mt][1][0], accdv[mt][1][1], accdv[mt][1][2], accdv[mt][1][3],
                     amv[0], amv[1], amv[2], amv[3], bd2[0], bd2[1]);
            mma_bf16(accdk[mt][0][0], accdk[mt][0][1], accdk[mt][0][2], accdk[mt][0][3],
                     ams[0], ams[1], ams[2], ams[3], bq[0], bq[1]);
            mma_bf16(accdk[mt][1][0], accdk[mt][1][1], accdk[mt][1][2], accdk[mt][1][3],
                     ams[0], ams[1], ams[2], ams[3], bq2[0], bq2[1]);
        }
        __syncthreads();
    }

    // Kt/Vt are dead after the q loop; reuse as staging for full-row stores
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        int lr = mt * 16 + (lane >> 2);
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            int c0 = warp * 16 + nt * 8 + (lane & 3) * 2;
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                *(__nv_bfloat162*)(Vt + (lr + hh * 8) * ATTN_D + c0) =
                    __float22bfloat162_rn(make_float2(accdv[mt][nt][hh * 2], accdv[mt][nt][hh * 2 + 1]));
                *(__nv_bfloat162*)(Kt + (lr + hh * 8) * ATTN_D + c0) =
                    __float22bfloat162_rn(make_float2(accdk[mt][nt][hh * 2] * scale, accdk[mt][nt][hh * 2 + 1] * scale));
            }
        }
    }
    __syncthreads();
    bf16* dkb = dk + ((long)b * S) * hb + h * ATTN_D;
    bf16* dvb = dv + ((long)b * S) * hb + h * ATTN_D;
    for (int idx = tid; idx < 32 * (ATTN_D / 8); idx += 128) {
        int r = idx >> 3, c8 = idx & 7;
        int gr = s0 + r;
        if (gr >= S) continue;
        *(uint4*)(dvb + (long)gr * hb + c8 * 8) = *(uint4*)(Vt + r * ATTN_D + c8 * 8);
        *(uint4*)(dkb + (long)gr * hb + c8 * 8) = *(uint4*)(Kt + r * ATTN_D + c8 * 8);
    }
}

static int sp_of(int S) {
    int sp = (S + 15) & ~15;
    return sp <= 32 ? 32 : sp <= 64 ? 64 : sp <= 96 ? 96 : 128;
}

#define ATTN_DISPATCH(SP, ...) \
    switch (SP) { \
        case 32: { __VA_ARGS__(32); break; } \
        case 64: { __VA_ARGS__(64); break; } \
        case 96: { __VA_ARGS__(96); break; } \
        default: { __VA_ARGS__(128); break; } \
    }


static inline void attn_fwd_launch(const bf16* q, const bf16* k, const bf16* v, bf16* o, float* lse,
                                   int B, int S, int H, float scale, cudaStream_t st) {
    int sp = sp_of(S);
    const int smem = 3 * sp * ATTN_D * 2;
#define FWD_CASE(SPV) \
    { \
        static bool cfgd = false; \
        if (!cfgd) { \
            cudaFuncSetAttribute(attn_fwd_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
            cfgd = true; \
        } \
        attn_fwd_kernel<SPV><<<B * H, 128, smem, st>>>(q, k, v, o, lse, S, H, scale, \
            (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, \
            (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D); \
    }
    ATTN_DISPATCH(sp, FWD_CASE);
#undef FWD_CASE
}

static inline void attn_bwd_launch(const bf16* q, const bf16* k, const bf16* v, const bf16* o,
                                   const bf16* dout, const float* lse, bf16* dq, bf16* dk, bf16* dv,
                                   float* dsum, int B, int S, int H, float scale, cudaStream_t st) {
    int sp = sp_of(S);
    const int smem1 = 4 * sp * ATTN_D * 2;
    const int smem2 = (2 * 32 * ATTN_D + 6 * 16 * ATTN_D + 2 * 16 * 40) * 2;
#define BWD_CASE(SPV) \
    { \
        static bool cfgd1 = false, cfgd2 = false; \
        if (!cfgd1) { \
            cudaFuncSetAttribute(attn_bwd_dq_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem1); \
            cfgd1 = true; \
        } \
        if (!cfgd2) { \
            cudaFuncSetAttribute(attn_bwd_dkdv_kernel<SPV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem2); \
            cfgd2 = true; \
        } \
        attn_bwd_dq_kernel<SPV><<<B * H, 128, smem1, st>>>(q, k, v, o, dout, lse, dq, dsum, S, H, scale, \
            (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, \
            (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D); \
        dim3 g2(sp / 32, B * H); \
        attn_bwd_dkdv_kernel<SPV><<<g2, 128, smem2, st>>>(q, k, v, dout, lse, dsum, dk, dv, S, H, scale, \
            (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, (long)S * H * ATTN_D, \
            (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D, (long)H * ATTN_D); \
    }
    ATTN_DISPATCH(sp, BWD_CASE);
#undef BWD_CASE
}
