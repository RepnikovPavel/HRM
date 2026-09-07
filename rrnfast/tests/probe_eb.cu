#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "../csrc/edge_bwd.cu"

std::vector<torch::Tensor> run_probe(
    torch::Tensor hw12, torch::Tensor nb, torch::Tensor pos, torch::Tensor b1,
    torch::Tensor w2, torch::Tensor b2, torch::Tensor w3, torch::Tensor b3,
    torch::Tensor w4, torch::Tensor b4, torch::Tensor dm, int64_t B) {
    auto f32 = torch::TensorOptions().dtype(torch::kFloat).device(hw12.device());
    auto ws2 = torch::zeros({96, 96}, f32);
    auto ws3 = torch::zeros({96, 96}, f32);
    auto ws4 = torch::zeros({96, 96}, f32);
    auto db1 = torch::zeros({96}, f32);
    auto db2 = torch::zeros({96}, f32);
    auto db3 = torch::zeros({96}, f32);
    auto db4 = torch::zeros({96}, f32);
    auto dz0 = torch::empty({B, 81, 20, 96}, hw12.options());
    auto dhw12 = torch::empty({B * 81, 192}, hw12.options());
    auto w2t = w2.t().contiguous();
    auto w3t = w3.t().contiguous();
    auto w4t = w4.t().contiguous();
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    edge_bwd_launch((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                    (const int*)pos.data_ptr(), (const float*)b1.data_ptr(),
                    (const bf16*)w2.data_ptr(), (const bf16*)b2.data_ptr(),
                    (const bf16*)w3.data_ptr(), (const bf16*)b3.data_ptr(),
                    (const bf16*)w4.data_ptr(), (const bf16*)b4.data_ptr(),
                    (const bf16*)w2t.data_ptr(), (const bf16*)w3t.data_ptr(),
                    (const bf16*)w4t.data_ptr(),
                    (const bf16*)dm.data_ptr() + 96,
                    (float*)ws2.data_ptr(), (float*)ws3.data_ptr(), (float*)ws4.data_ptr(),
                    (float*)db1.data_ptr(), (float*)db2.data_ptr(),
                    (float*)db3.data_ptr(), (float*)db4.data_ptr(),
                    (bf16*)dz0.data_ptr(), (bf16*)dhw12.data_ptr(),
                    (int)B, 0, 0.f, 0.f, 0, 0, sm_count,
                    at::cuda::getCurrentCUDAStream());
    return {ws2, ws3, ws4, db1, db2, db3, db4, dz0, dhw12};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run_probe", &run_probe); }
