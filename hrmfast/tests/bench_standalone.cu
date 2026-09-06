#include "../csrc/gemm_family.cuh"
#include <cstdio>
#include <cstdlib>

int main() {
    int M = 90304, K = 512, N = 1536;
    bf16 *x, *wg, *wu, *y;
    cudaMalloc(&x, (long)M * K * 2);
    cudaMalloc(&wg, (long)N * K * 2);
    cudaMalloc(&wu, (long)N * K * 2);
    cudaMalloc(&y, (long)M * N * 2);
    cudaMemset(x, 0x3c, (long)M * K * 2);
    cudaMemset(wg, 0x3c, (long)N * K * 2);
    cudaMemset(wu, 0x3c, (long)N * K * 2);

    for (int i = 0; i < 5; ++i)
        tn_launch<1, 2>(0, x, wg, wu, nullptr, y, nullptr, nullptr, M, N, K, 0);
    cudaDeviceSynchronize();
    printf("err=%d\n", (int)cudaGetLastError());

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int n = 50;
    for (int i = 0; i < n; ++i)
        tn_launch<1, 2>(0, x, wg, wu, nullptr, y, nullptr, nullptr, M, N, K, 0);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= n;
    double fl = 2.0 * M * N * K * 2;
    printf("kernel: %.3f ms, %.1f TFLOPS\n", ms, fl / (ms * 1e-3) / 1e12);
    return 0;
}
