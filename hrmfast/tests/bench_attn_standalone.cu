#include "../csrc/attention_kernels.cuh"
#include <cstdio>
#include <cstdlib>

int main() {
    int B = 1088, S = 82, H = 8;
    const long n = (long)B * S * H * ATTN_D;
    bf16 *q, *k, *v, *o, *dout, *dq, *dk, *dv;
    float *lse, *dsum;
    cudaMalloc(&q, n * 2); cudaMalloc(&k, n * 2); cudaMalloc(&v, n * 2);
    cudaMalloc(&o, n * 2); cudaMalloc(&dout, n * 2);
    cudaMalloc(&dq, n * 2); cudaMalloc(&dk, n * 2); cudaMalloc(&dv, n * 2);
    cudaMalloc(&lse, (long)B * H * S * 4); cudaMalloc(&dsum, (long)B * H * S * 4);
    cudaMemset(q, 0x3c, n * 2); cudaMemset(k, 0x3c, n * 2); cudaMemset(v, 0x3c, n * 2);
    cudaMemset(dout, 0x3c, n * 2);
    float sc = 0.125f;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int i = 0; i < 5; ++i) attn_fwd_launch(q, k, v, o, lse, B, S, H, sc, 0);
    cudaDeviceSynchronize();
    printf("fwd err=%d\n", (int)cudaGetLastError());
    cudaEventRecord(t0);
    int iters = 50;
    for (int i = 0; i < iters; ++i) attn_fwd_launch(q, k, v, o, lse, B, S, H, sc, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms; cudaEventElapsedTime(&ms, t0, t1);
    printf("attn fwd: %.3f ms\n", ms / iters);

    for (int i = 0; i < 5; ++i) attn_bwd_launch(q, k, v, o, dout, lse, dq, dk, dv, dsum, B, S, H, sc, 0);
    cudaDeviceSynchronize();
    printf("bwd err=%d\n", (int)cudaGetLastError());
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) attn_bwd_launch(q, k, v, o, dout, lse, dq, dk, dv, dsum, B, S, H, sc, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms, t0, t1);
    printf("attn bwd: %.3f ms\n", ms / iters);
    return 0;
}
