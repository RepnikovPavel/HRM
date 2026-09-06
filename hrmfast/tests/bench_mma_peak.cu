#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(256, 1) mma_peak(float* out, int iters) {
    float acc[8][4] = {};
    uint32_t a[4], b[2];
    a[0] = threadIdx.x; a[1] = threadIdx.x + 1; a[2] = 2; a[3] = 3;
    b[0] = threadIdx.x; b[1] = 5;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            mma_bf16(acc[j][0], acc[j][1], acc[j][2], acc[j][3], a[0], a[1], a[2], a[3], b[0], b[1]);
    }
    float s = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][1] + acc[j][2] + acc[j][3];
    if (s == -1.f) out[threadIdx.x] = s;
}

int main() {
    float* out;
    cudaMalloc(&out, 1024);
    int iters = 20000;
    dim3 grid(36 * 4);
    mma_peak<<<grid, 256>>>(out, 10);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    int reps = 20;
    for (int r = 0; r < reps; ++r) mma_peak<<<grid, 256>>>(out, iters);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    ms /= reps;
    double flops = (double)grid.x * 256 / 32 * (double)iters * 8 * (16.0 * 8 * 16 * 2);
    printf("mma peak: %.3f ms, %.1f TFLOPS (err=%d)\n", ms, flops / (ms * 1e-3) / 1e12, (int)cudaGetLastError());
    return 0;
}
