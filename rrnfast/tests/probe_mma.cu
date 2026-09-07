#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cstdint>
using bf16 = __nv_bfloat16;

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A[m][k] = (m == M0 && k == K0) ? 1 : 0 -> out[M0][n] = B[K0][n] = W[n][K0]
// B loaded as b0 = {W[n][k], W[n][k+1]}, b1 = {W[n][k+8], W[n][k+9]}
__global__ void probe(const bf16* W, float* out, int M0, int K0) {
    int lane = threadIdx.x;
    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    uint32_t a[4] = {0, 0, 0, 0};
    // a-frag: a0/a1 = (row lane/4, k (lane%4)*2+{0,1}); a2/a3 = row+8; a4..a7 = k+8
    int kbase = (lane & 3) * 2;
    int idx = (K0 >= 8) ? 2 : 0;      // reg pair: k<8 -> a0/a1(a2/a3), k>=8 -> a4/a5...
    int kk = K0 & 7;
    int reg = idx + ((M0 >= 8) ? 1 : 0);
    if ((M0 & 7) == (lane >> 2) && kk / 2 == (lane & 3)) {
        __nv_bfloat162 one = __float22bfloat162_rn(
            make_float2((K0 & 1) ? 0.f : 1.f, (K0 & 1) ? 1.f : 0.f));
        a[reg] = *(uint32_t*)&one;
    }
    int n = lane >> 2;
    const bf16* p = W + n * 16;
    uint32_t b0 = *(const uint32_t*)(p + (lane & 3) * 2);
    uint32_t b1 = *(const uint32_t*)(p + (lane & 3) * 2 + 8);
    mma_bf16(c0, c1, c2, c3, a[0], a[1], a[2], a[3], b0, b1);
    int row = lane >> 2, col = (lane & 3) * 2;
    out[row * 8 + col] = c0;
    out[row * 8 + col + 1] = c1;
    out[(row + 8) * 8 + col] = c2;
    out[(row + 8) * 8 + col + 1] = c3;
}

torch::Tensor run_probe(torch::Tensor W, int64_t m0, int64_t k0) {
    auto out = torch::zeros({16, 8}, W.options().dtype(torch::kFloat));
    probe<<<1, 32>>>((const bf16*)W.data_ptr(), (float*)out.data_ptr(), (int)m0, (int)k0);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
