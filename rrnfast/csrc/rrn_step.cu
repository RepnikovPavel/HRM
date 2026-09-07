#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "edge_common.cuh"

void edge_fwd_launch(const bf16*, const int*, const float*,
                     const bf16*, const bf16*, const bf16*, const bf16*,
                     const bf16*, const bf16*, bf16*,
                     int, int, float, float, uint64_t, int, cudaStream_t);
void edge_bwd_launch(const bf16*, const int*, const int*, const float*,
                     const bf16*, const bf16*, const bf16*, const bf16*,
                     const bf16*, const bf16*,
                     const bf16*, const bf16*, const bf16*, const bf16*,
                     float*, float*, float*, float*, float*, float*, float*,
                     bf16*, bf16*, int, int, float, float, uint64_t, int, int, cudaStream_t);
void lstm_fwd_launch(const bf16*, const bf16*, bf16*, bf16*, long, cudaStream_t);
void lstm_bwd_launch(const bf16*, const bf16*, const bf16*, const bf16*, bf16*, bf16*, long,
                     cudaStream_t);

struct StepW {
    torch::Tensor w12, w2, b2, w3, b3, w4, b4, wls, wih, whh, b1f;
    torch::Tensor w2t, w3t, w4t;
};

static StepW prep_weights(torch::Tensor& w0, torch::Tensor& b1,
                          torch::Tensor& w2, torch::Tensor& b2,
                          torch::Tensor& w3, torch::Tensor& b3,
                          torch::Tensor& w4, torch::Tensor& b4,
                          torch::Tensor& wih, torch::Tensor& whh) {
    auto bf = torch::kBFloat16;
    StepW s;
    auto w0b = w0.to(bf);
    s.w12 = torch::cat({w0b.narrow(1, 0, 96), w0b.narrow(1, 96, 96)}, 0).contiguous();
    s.w2 = w2.to(bf).contiguous();
    s.b2 = b2.to(bf).contiguous();
    s.w3 = w3.to(bf).contiguous();
    s.b3 = b3.to(bf).contiguous();
    s.w4 = w4.to(bf).contiguous();
    s.b4 = b4.to(bf).contiguous();
    s.w2t = s.w2.t().contiguous();
    s.w3t = s.w3.t().contiguous();
    s.w4t = s.w4.t().contiguous();
    s.wih = wih.to(bf).contiguous();
    s.whh = whh.to(bf).contiguous();
    s.wls = torch::cat({s.wih, s.whh}, 1).contiguous();
    s.b1f = b1.to(torch::kFloat).contiguous();
    return s;
}

static void check_step_inputs(torch::Tensor& x, torch::Tensor& h, torch::Tensor& c) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kBFloat16);
    TORCH_CHECK(h.is_contiguous() && c.is_contiguous() && x.is_contiguous());
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 81 && x.size(2) == 96);
}

static void check_nb(torch::Tensor& nb) {
    TORCH_CHECK(nb.scalar_type() == torch::kInt && nb.is_contiguous());
}

std::vector<torch::Tensor> rrn_step_fwd(
    torch::Tensor x, torch::Tensor h, torch::Tensor c, torch::Tensor nb,
    torch::Tensor w0, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3, torch::Tensor w4, torch::Tensor b4,
    torch::Tensor wih, torch::Tensor whh,
    bool train, double p, int64_t seed, int64_t step)
{
    check_step_inputs(x, h, c);
    check_nb(nb);
    int B = x.size(0);
    long M = (long)B * 81;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto bf = x.options();
    StepW s = prep_weights(w0, b1, w2, b2, w3, b3, w4, b4, wih, whh);

    auto hv = h.view({M, 96});
    auto hw12 = torch::empty({M, 192}, bf);
    tn_launch<0, 1>(0, (const bf16*)hv.data_ptr(), (const bf16*)s.w12.data_ptr(), nullptr,
                    nullptr, (bf16*)hw12.data_ptr(), nullptr, nullptr, M, 192, 96, stream);

    auto m = torch::empty({B, 81, 96}, bf);
    float scale = train ? (float)(1.0 / (1.0 - p)) : 0.f;
    edge_fwd_launch((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                    (const float*)s.b1f.data_ptr(),
                    (const bf16*)s.w2.data_ptr(), (const bf16*)s.b2.data_ptr(),
                    (const bf16*)s.w3.data_ptr(), (const bf16*)s.b3.data_ptr(),
                    (const bf16*)s.w4.data_ptr(), (const bf16*)s.b4.data_ptr(),
                    (bf16*)m.data_ptr(), B, train ? 1 : 0, (float)p, scale,
                    (uint64_t)seed, (int)step, stream);

    auto xm2 = torch::empty({M, 288}, bf);
    xm2.narrow(1, 0, 96).copy_(x.view({M, 96}));
    xm2.narrow(1, 96, 96).copy_(m.view({M, 96}));
    xm2.narrow(1, 192, 96).copy_(hv);

    auto gates = torch::empty({M, 384}, bf);
    tn_launch<0, 1>(0, (const bf16*)xm2.data_ptr(), (const bf16*)s.wls.data_ptr(), nullptr,
                    nullptr, (bf16*)gates.data_ptr(), nullptr, nullptr, M, 384, 288, stream);

    auto h1 = torch::empty({M, 96}, bf);
    auto c1 = torch::empty({M, 96}, bf);
    lstm_fwd_launch((const bf16*)gates.data_ptr(), (const bf16*)c.data_ptr(),
                    (bf16*)h1.data_ptr(), (bf16*)c1.data_ptr(), M, stream);
    return {h1.view({B, 81, 96}), c1.view({B, 81, 96}), m};
}

std::vector<torch::Tensor> rrn_step_bwd(
    torch::Tensor x, torch::Tensor h, torch::Tensor c, torch::Tensor m,
    torch::Tensor dh, torch::Tensor dc, torch::Tensor nb, torch::Tensor pos,
    torch::Tensor w0, torch::Tensor b1, torch::Tensor w2, torch::Tensor b2,
    torch::Tensor w3, torch::Tensor b3, torch::Tensor w4, torch::Tensor b4,
    torch::Tensor wih, torch::Tensor whh,
    bool train, double p, int64_t seed, int64_t step)
{
    check_nb(nb);
    TORCH_CHECK(pos.scalar_type() == torch::kInt && pos.is_contiguous());
    int B = x.size(0);
    long M = (long)B * 81;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto bf = x.options();
    auto f32 = x.options().dtype(torch::kFloat);
    StepW s = prep_weights(w0, b1, w2, b2, w3, b3, w4, b4, wih, whh);

    auto xm2 = torch::empty({M, 288}, bf);
    xm2.narrow(1, 0, 96).copy_(x.view({M, 96}));
    xm2.narrow(1, 96, 96).copy_(m.view({M, 96}));
    xm2.narrow(1, 192, 96).copy_(h.view({M, 96}));

    auto gates = torch::empty({M, 384}, bf);
    tn_launch<0, 1>(0, (const bf16*)xm2.data_ptr(), (const bf16*)s.wls.data_ptr(), nullptr,
                    nullptr, (bf16*)gates.data_ptr(), nullptr, nullptr, M, 384, 288, stream);

    auto dgates = torch::empty({M, 384}, bf);
    auto dcp = torch::empty({M, 96}, bf);
    lstm_bwd_launch((const bf16*)gates.data_ptr(), (const bf16*)c.data_ptr(),
                    (const bf16*)dh.data_ptr(), (const bf16*)dc.data_ptr(),
                    (bf16*)dgates.data_ptr(), (bf16*)dcp.data_ptr(), M, stream);

    auto dxm = torch::empty({M, 192}, bf);
    nn_launch<1>(0, (const bf16*)dgates.data_ptr(), nullptr, (const bf16*)s.wih.data_ptr(),
                 nullptr, (bf16*)dxm.data_ptr(), M, 384, 192, stream);

    auto ws_ls = torch::zeros({384, 288}, f32);
    nt_launch<1>(0, (const bf16*)dgates.data_ptr(), nullptr, (const bf16*)xm2.data_ptr(),
                 (float*)ws_ls.data_ptr(), nullptr, M, 384, 288, stream);

    auto dh_lstm = torch::empty({M, 96}, bf);
    nn_launch<1>(0, (const bf16*)dgates.data_ptr(), nullptr, (const bf16*)s.whh.data_ptr(),
                 nullptr, (bf16*)dh_lstm.data_ptr(), M, 384, 96, stream);

    auto hw12 = torch::empty({M, 192}, bf);
    tn_launch<0, 1>(0, (const bf16*)h.data_ptr(), (const bf16*)s.w12.data_ptr(), nullptr,
                    nullptr, (bf16*)hw12.data_ptr(), nullptr, nullptr, M, 192, 96, stream);

    auto dz0 = torch::empty({B, 81, 20, 96}, bf);
    auto ws2 = torch::zeros({96, 96}, f32);
    auto ws3 = torch::zeros({96, 96}, f32);
    auto ws4 = torch::zeros({96, 96}, f32);
    auto db1 = torch::zeros({96}, f32);
    auto db2 = torch::zeros({96}, f32);
    auto db3 = torch::zeros({96}, f32);
    auto db4 = torch::zeros({96}, f32);
    auto dhw12 = torch::empty({M, 192}, bf);

    float scale = train ? (float)(1.0 / (1.0 - p)) : 0.f;
    int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const bf16* dm_ptr = (const bf16*)dxm.data_ptr() + 96;
    edge_bwd_launch((const bf16*)hw12.data_ptr(), (const int*)nb.data_ptr(),
                    (const int*)pos.data_ptr(), (const float*)s.b1f.data_ptr(),
                    (const bf16*)s.w2.data_ptr(), (const bf16*)s.b2.data_ptr(),
                    (const bf16*)s.w3.data_ptr(), (const bf16*)s.b3.data_ptr(),
                    (const bf16*)s.w4.data_ptr(), (const bf16*)s.b4.data_ptr(),
                    (const bf16*)s.w2t.data_ptr(), (const bf16*)s.w3t.data_ptr(),
                    (const bf16*)s.w4t.data_ptr(), dm_ptr,
                    (float*)ws2.data_ptr(), (float*)ws3.data_ptr(), (float*)ws4.data_ptr(),
                    (float*)db1.data_ptr(), (float*)db2.data_ptr(),
                    (float*)db3.data_ptr(), (float*)db4.data_ptr(),
                    (bf16*)dz0.data_ptr(), (bf16*)dhw12.data_ptr(),
                    B, train ? 1 : 0, (float)p, scale, (uint64_t)seed, (int)step,
                    sm_count, stream);

    auto ws12 = torch::zeros({192, 96}, f32);
    nt_launch<1>(0, (const bf16*)dhw12.data_ptr(), nullptr, (const bf16*)h.data_ptr(),
                 (float*)ws12.data_ptr(), nullptr, M, 192, 96, stream);

    auto dh_edge = torch::empty({M, 96}, bf);
    nn_launch<1>(0, (const bf16*)dhw12.data_ptr(), nullptr, (const bf16*)s.w12.data_ptr(),
                 nullptr, (bf16*)dh_edge.data_ptr(), M, 192, 96, stream);

    auto dhp = (dh_lstm + dh_edge).view({B, 81, 96});
    auto dx = dxm.narrow(1, 0, 96).contiguous().view({B, 81, 96});
    auto dw0 = torch::cat({ws12.narrow(0, 0, 96), ws12.narrow(0, 96, 96)}, 1).contiguous();
    auto dwih = ws_ls.narrow(1, 0, 192).contiguous();
    auto dwhh = ws_ls.narrow(1, 192, 96).contiguous();
    return {dx, dhp, dcp.view({B, 81, 96}), dw0, db1, ws2, db2, ws3, db3, ws4, db4, dwih, dwhh};
}
