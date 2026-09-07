import torch
import rrnfast_backend as _be


def neighbor_pos(neighbors):
    nb = neighbors.cpu()
    pos = torch.zeros(81, 20, dtype=torch.int32)
    for s in range(81):
        for k in range(20):
            j = int(nb[s, k])
            pos[s, k] = int((nb[j] == s).nonzero()[0, 0])
    return pos.to(neighbors.device)


class RRNStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h, c, nb, pos, w0, b1, w2, b2, w3, b3, w4, b4, wih, whh,
                train, p, seed, step):
        h1, c1, m = _be.rrn_step_fwd(x, h, c, nb, w0, b1, w2, b2, w3, b3, w4, b4,
                                     wih, whh, train, p, seed, step)
        ctx.save_for_backward(x, h, c, m, nb, pos, w0, b1, w2, b2, w3, b3, w4, b4, wih, whh)
        ctx.misc = (train, p, seed, step)
        return h1, c1

    @staticmethod
    def backward(ctx, dh1, dc1):
        (x, h, c, m, nb, pos, w0, b1, w2, b2, w3, b3, w4, b4, wih, whh) = ctx.saved_tensors
        train, p, seed, step = ctx.misc
        dh1 = dh1.contiguous()
        dc1 = dc1.contiguous() if dc1 is not None else torch.zeros_like(c)
        out = _be.rrn_step_bwd(x, h, c, m, dh1, dc1, nb, pos, w0, b1, w2, b2, w3, b3,
                               w4, b4, wih, whh, train, p, seed, step)
        dx, dhp, dcp = out[0], out[1], out[2]
        dws = out[3:]
        return (dx, dhp, dcp, None, None, *dws, None, None, None, None)


def model_weights(model):
    ml = model.msg_layer
    return (ml[0].weight, ml[0].bias, ml[2].weight, ml[2].bias,
            ml[4].weight, ml[4].bias, ml[6].weight, ml[6].bias,
            model.lstm_ih.weight, model.lstm_hh.weight)


def rrn_step(x, h, c, model, train, seed=0, step=0, nb=None, pos=None):
    if nb is None:
        nb = model._rrnfast_nb
    if pos is None:
        pos = model._rrnfast_pos
    w = model_weights(model)
    return RRNStep.apply(x, h, c, nb, pos, *w, train, model.edge_drop if train else 0.0,
                         seed, step)


def prepare(model):
    model._rrnfast_pos = neighbor_pos(model.neighbors)
    model._rrnfast_nb = model.neighbors.int()


def run_steps(x, h, c, model, num_steps, train, seed=0):
    nb = model.neighbors
    pos = getattr(model, "_rrnfast_pos", None)
    if pos is None:
        pos = neighbor_pos(nb)
    for t in range(num_steps):
        h, c = rrn_step(x, h, c, model, train, seed=seed, step=t, nb=nb, pos=pos)
    return h, c
