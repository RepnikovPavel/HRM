#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "edge_common.cuh"

// rounding after every op mirrors the reference's bf16 TensorIterator ops so
// parity vs SudokuRRN._step stays at ulp level instead of accumulating drift
__device__ __forceinline__ float sig_bf16(float g) {
    return __bfloat162float(__float2bfloat16(1.f / (1.f + expf(-g))));
}

__global__ void lstm_fwd_kernel(const bf16* __restrict__ gates, const bf16* __restrict__ c,
                                bf16* __restrict__ h1, bf16* __restrict__ c1, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long row = idx / 12, cb = idx % 12;
    const bf16* gi = gates + row * 384 + cb * 8;
    uint4 vi = *(const uint4*)gi;
    uint4 vf = *(const uint4*)(gi + 96);
    uint4 vg = *(const uint4*)(gi + 192);
    uint4 vo = *(const uint4*)(gi + 288);
    uint4 vc = *(const uint4*)(c + row * 96 + cb * 8);
    const bf16 *pi = (const bf16*)&vi, *pf = (const bf16*)&vf, *pg = (const bf16*)&vg,
               *po = (const bf16*)&vo, *pc = (const bf16*)&vc;
    uint4 oh, oc;
    bf16* ph = (bf16*)&oh;
    bf16* pn = (bf16*)&oc;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float iv = sig_bf16(__bfloat162float(pi[i]));
        float fv = sig_bf16(__bfloat162float(pf[i]));
        float gv = __bfloat162float(__float2bfloat16(tanhf(__bfloat162float(pg[i]))));
        float ov = sig_bf16(__bfloat162float(po[i]));
        float t1 = bf16_round(fv * __bfloat162float(pc[i]));
        float t2 = bf16_round(iv * gv);
        float cn = bf16_round(t1 + t2);
        float th = bf16_round(tanhf(cn));
        ph[i] = __float2bfloat16(ov * th);
        pn[i] = __float2bfloat16(cn);
    }
    *(uint4*)(h1 + row * 96 + cb * 8) = oh;
    *(uint4*)(c1 + row * 96 + cb * 8) = oc;
}

__global__ void lstm_bwd_kernel(const bf16* __restrict__ gates, const bf16* __restrict__ c,
                                const bf16* __restrict__ dh, const bf16* __restrict__ dc,
                                bf16* __restrict__ dgates, bf16* __restrict__ dcp, long total) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long row = idx / 12, cb = idx % 12;
    const bf16* gi = gates + row * 384 + cb * 8;
    uint4 vi = *(const uint4*)gi;
    uint4 vf = *(const uint4*)(gi + 96);
    uint4 vg = *(const uint4*)(gi + 192);
    uint4 vo = *(const uint4*)(gi + 288);
    uint4 vc = *(const uint4*)(c + row * 96 + cb * 8);
    uint4 vh = *(const uint4*)(dh + row * 96 + cb * 8);
    uint4 vn = *(const uint4*)(dc + row * 96 + cb * 8);
    const bf16 *pi = (const bf16*)&vi, *pf = (const bf16*)&vf, *pg = (const bf16*)&vg,
               *po = (const bf16*)&vo, *pc = (const bf16*)&vc, *pd = (const bf16*)&vh,
               *pn = (const bf16*)&vn;
    uint4 oi, of, og, oo, od;
    bf16 *qi = (bf16*)&oi, *qf = (bf16*)&of, *qg = (bf16*)&og, *qo = (bf16*)&oo,
         *qd = (bf16*)&od;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        float iv = sig_bf16(__bfloat162float(pi[i]));
        float fv = sig_bf16(__bfloat162float(pf[i]));
        float gv = __bfloat162float(__float2bfloat16(tanhf(__bfloat162float(pg[i]))));
        float ov = sig_bf16(__bfloat162float(po[i]));
        float cv = __bfloat162float(pc[i]);
        float cn = bf16_round(bf16_round(fv * cv) + bf16_round(iv * gv));
        float th = bf16_round(tanhf(cn));
        float dhv = __bfloat162float(pd[i]);
        float dov = bf16_round(dhv * th);
        float dth = bf16_round(dhv * ov);
        float dch = bf16_round(dth * bf16_round(1.f - th * th));
        float dct = bf16_round(__bfloat162float(pn[i]) + dch);
        float dfv = bf16_round(dct * cv);
        float dcv = bf16_round(dct * fv);
        float div = bf16_round(dct * gv);
        float dgv = bf16_round(dct * iv);
        qi[i] = __float2bfloat16(bf16_round(div * bf16_round((1.f - iv) * iv)));
        qf[i] = __float2bfloat16(bf16_round(dfv * bf16_round((1.f - fv) * fv)));
        qg[i] = __float2bfloat16(bf16_round(dgv * bf16_round(1.f - gv * gv)));
        qo[i] = __float2bfloat16(bf16_round(dov * bf16_round((1.f - ov) * ov)));
        qd[i] = __float2bfloat16(dcv);
    }
    bf16* go = dgates + row * 384 + cb * 8;
    *(uint4*)go = oi;
    *(uint4*)(go + 96) = of;
    *(uint4*)(go + 192) = og;
    *(uint4*)(go + 288) = oo;
    *(uint4*)(dcp + row * 96 + cb * 8) = od;
}

void lstm_fwd_launch(const bf16* gates, const bf16* c, bf16* h1, bf16* c1, long M,
                     cudaStream_t stream) {
    long total = M * 12;
    int blocks = (int)((total + 255) / 256);
    lstm_fwd_kernel<<<blocks, 256, 0, stream>>>(gates, c, h1, c1, total);
}

void lstm_bwd_launch(const bf16* gates, const bf16* c, const bf16* dh, const bf16* dc,
                     bf16* dgates, bf16* dcp, long M, cudaStream_t stream) {
    long total = M * 12;
    int blocks = (int)((total + 255) / 256);
    lstm_bwd_kernel<<<blocks, 256, 0, stream>>>(gates, c, dh, dc, dgates, dcp, total);
}
