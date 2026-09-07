#include <torch/extension.h>

std::vector<torch::Tensor> rrn_step_fwd(
    torch::Tensor x, torch::Tensor h, torch::Tensor c, torch::Tensor nb,
    torch::Tensor w0, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3, torch::Tensor w4, torch::Tensor b4,
    torch::Tensor wih, torch::Tensor whh,
    bool train, double p, int64_t seed, int64_t step);

std::vector<torch::Tensor> rrn_step_bwd(
    torch::Tensor x, torch::Tensor h, torch::Tensor c, torch::Tensor m,
    torch::Tensor dh, torch::Tensor dc, torch::Tensor nb, torch::Tensor pos,
    torch::Tensor w0, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3, torch::Tensor w4, torch::Tensor b4,
    torch::Tensor wih, torch::Tensor whh,
    bool train, double p, int64_t seed, int64_t step);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rrn_step_fwd", &rrn_step_fwd, "one fused RRN step forward -> (h1, c1, m)");
    m.def("rrn_step_bwd", &rrn_step_bwd, "one fused RRN step backward");
}
