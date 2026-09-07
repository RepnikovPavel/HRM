#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../csrc/gemm_family.cuh"

torch::Tensor tn(torch::Tensor a, torch::Tensor b) {
    int M = a.size(0), K = a.size(1), N = b.size(0);
    auto out = torch::empty({M, N}, a.options());
    tn_launch<0, 1>(0, (const bf16*)a.data_ptr(), (const bf16*)b.data_ptr(), nullptr,
                    nullptr, (bf16*)out.data_ptr(), nullptr, nullptr, M, N, K,
                    at::cuda::getCurrentCUDAStream());
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tn", &tn); }
