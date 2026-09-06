import torch
import hrmfast_backend as _be

# tile config ids measured on 2x RTX 5060 Ti (sm_120) by hrmfast/tests/sweep_gemm.py
# at M=90304 (ms, best per op/shape):
#   tn 1536x512: cfg0 2.917 | tn 512x512: cfg0 0.994 | tn 512x1536: cfg3 2.871
#   nn 1536x512: cfg3 2.897 | nn 512x512: cfg0 1.003 | nn 512x1536: cfg0 3.047
#   nt 1536x512: cfg2 3.007 | nt 512x512: cfg2 1.093 | nt 512x1536: cfg2 2.998
#   swiglu 1536x512: cfg1 5.897 (cfg0 5.930)
# key = (op, N, K), value = cfg id into HRMFAST_CFGS in csrc/gemm_family.cuh
_GEMM_CFG = {
    ("tn", 1536, 512): 0,
    ("tn", 512, 512): 0,
    ("tn", 512, 1536): 3,
    ("swiglu", 1536, 512): 1,
    ("nn", 1536, 512): 3,
    ("nn", 512, 512): 0,
    ("nn", 512, 1536): 0,
    ("nt", 1536, 512): 2,
    ("nt", 512, 512): 2,
    ("nt", 512, 1536): 2,
}


def _cfgs(N, K):
    return (_GEMM_CFG.get(("tn", N, K), 0), _GEMM_CFG.get(("nn", N, K), 0),
            _GEMM_CFG.get(("nt", N, K), 0))


def swiglu_forward(x, w_gate, w_up):
    return _be.swiglu_forward(x, w_gate, w_up, _GEMM_CFG.get(("swiglu", w_gate.shape[0], x.shape[1]), 0))


class SwiGLUFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_gate, w_up):
        y = swiglu_forward(x, w_gate, w_up)
        ctx.save_for_backward(x, w_gate, w_up)
        return y

    @staticmethod
    def backward(ctx, dout):
        x, w_gate, w_up = ctx.saved_tensors
        N, K = w_gate.shape[0], x.shape[1]
        cfg_tn = _GEMM_CFG.get(("swiglu", N, K), 0)
        cfg_nn = _GEMM_CFG.get(("nn2", N, K), _GEMM_CFG.get(("nn", N, K), 0))
        cfg_nt = _GEMM_CFG.get(("nt2", N, K), _GEMM_CFG.get(("nt", N, K), 0))
        dx, dw_gate, dw_up = _be.swiglu_backward(x, w_gate, w_up, dout.contiguous(),
                                                 cfg_tn, cfg_nn, cfg_nt)
        return dx, dw_gate, dw_up


def swiglu_fused(x, w_gate, w_up):
    return SwiGLUFused.apply(x, w_gate, w_up)


class LinearFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        y = _be.linear_fwd(x, w, _GEMM_CFG.get(("tn", w.shape[0], x.shape[1]), 0))
        ctx.save_for_backward(x, w)
        return y

    @staticmethod
    def backward(ctx, dout):
        x, w = ctx.saved_tensors
        N, K = w.shape[0], x.shape[1]
        dx, dw = _be.linear_backward(dout.contiguous(), x, w,
                                     _GEMM_CFG.get(("nn", N, K), 0), _GEMM_CFG.get(("nt", N, K), 0))
        return dx, dw


class LinearResidRmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, resid, eps):
        y, rstd = _be.linear_resid_rmsnorm_fwd(x, w, resid, eps,
                                               _GEMM_CFG.get(("tn", w.shape[0], x.shape[1]), 0))
        ctx.save_for_backward(y, rstd, x, w)
        return y

    @staticmethod
    def backward(ctx, dout):
        y, rstd, x, w = ctx.saved_tensors
        N, K = w.shape[0], x.shape[1]
        dx, dw, dh = _be.linear_resid_rmsnorm_backward(dout.contiguous(), y, rstd, x, w,
                                                       _GEMM_CFG.get(("nn", N, K), 0),
                                                       _GEMM_CFG.get(("nt", N, K), 0))
        return dx, dw, dh, None


def linear_resid_rmsnorm(x, w, resid, eps):
    return LinearResidRmsNorm.apply(x, w, resid, eps)


class AttentionFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):
        o, lse = _be.attn_forward(q, k, v, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, dout):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _be.attn_backward(q, k, v, o, dout.contiguous(), lse, ctx.scale)
        return dq, dk, dv, None


def attention_fused(q, k, v, scale):
    return AttentionFused.apply(q, k, v, scale)
