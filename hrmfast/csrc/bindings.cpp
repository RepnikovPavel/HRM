#include <torch/extension.h>

torch::Tensor swiglu_forward(torch::Tensor x, torch::Tensor w_gate, torch::Tensor w_up, int64_t cfg);
torch::Tensor linear_fwd(torch::Tensor x, torch::Tensor w, int64_t cfg);
std::vector<torch::Tensor> linear_resid_rmsnorm_fwd(torch::Tensor x, torch::Tensor w,
                                                    torch::Tensor resid, double eps, int64_t cfg);
std::vector<torch::Tensor> swiglu_backward(torch::Tensor x, torch::Tensor w_gate, torch::Tensor w_up,
                                           torch::Tensor dout, int64_t cfg_tn, int64_t cfg_nn, int64_t cfg_nt);
std::vector<torch::Tensor> linear_backward(torch::Tensor dout, torch::Tensor x, torch::Tensor w,
                                           int64_t cfg_nn, int64_t cfg_nt);
std::vector<torch::Tensor> linear_resid_rmsnorm_backward(torch::Tensor dy, torch::Tensor y,
                                                         torch::Tensor rstd, torch::Tensor x,
                                                         torch::Tensor w,
                                                         int64_t cfg_nn, int64_t cfg_nt);

torch::Tensor dbg_nn(torch::Tensor dy, torch::Tensor w, int64_t cfg);
torch::Tensor dbg_nt(torch::Tensor dy, torch::Tensor x, int64_t cfg);

std::vector<torch::Tensor> attn_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double scale);
std::vector<torch::Tensor> attn_backward(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                         torch::Tensor o, torch::Tensor dout, torch::Tensor lse,
                                         double scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swiglu_forward", &swiglu_forward, "y = silu(x @ Wg^T) * (x @ Wu^T)");
    m.def("linear_fwd", &linear_fwd, "y = x @ W^T (mma)");
    m.def("linear_resid_rmsnorm_fwd", &linear_resid_rmsnorm_fwd, "y = rmsnorm(resid + x @ W^T)");
    m.def("swiglu_backward", &swiglu_backward, "dx, dWg, dWu for fused SwiGLU");
    m.def("linear_backward", &linear_backward, "dx, dW for y = x @ W^T");
    m.def("linear_resid_rmsnorm_backward", &linear_resid_rmsnorm_backward, "backward for fused linear+resid+rmsnorm");
    m.def("dbg_nn", &dbg_nn, "dx only (sweep)");
    m.def("dbg_nt", &dbg_nt, "dW only (sweep)");
    m.def("attn_forward", &attn_forward, "fused attention fwd [B,S,H,64]");
    m.def("attn_backward", &attn_backward, "fused attention bwd");
}
