import argparse


def model_flops_per_step(batch, seq, hidden=512, inter=1536, heads=8, vocab=11,
                         h_cycles=2, l_cycles=2, h_layers=4, l_layers=4):
    block = 2 * hidden * (3 * hidden)      # qkv
    block += 4 * heads * seq * (hidden // heads)  # attention scores + values
    block += 2 * hidden * hidden           # o proj
    block += 2 * hidden * (2 * inter)      # gate_up
    block += 2 * inter * hidden            # down

    fwd_nograd = (h_cycles * l_cycles - 1) * l_layers + (h_cycles - 1) * h_layers
    fwd_grad = l_layers + h_layers
    module_passes = fwd_nograd + fwd_grad
    lm_head = 2 * hidden * vocab

    total_fwd = 2 * module_passes          # main + target-Q forward
    total_bwd = 2 * fwd_grad * 2           # backward dx + dW on grad passes
    flops = batch * seq * (block * (total_fwd + total_bwd) + 2 * lm_head)
    return flops


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, required=True, help="per-GPU batch size")
    p.add_argument("--seq", type=int, default=82)
    p.add_argument("--steps-per-sec", type=float, required=True)
    p.add_argument("--peak-tflops", type=float, default=52.6,
                   help="measured mma.sync ceiling of the GPU")
    args = p.parse_args()

    flops = model_flops_per_step(args.batch, args.seq)
    achieved = flops * args.steps_per_sec / 1e12
    print(f"model FLOPs/step/GPU: {flops / 1e12:.1f} TFLOP")
    print(f"achieved: {achieved:.1f} TFLOPS")
    print(f"compute efficiency: {achieved / args.peak_tflops * 100:.1f}% of {args.peak_tflops} TFLOPS ceiling")


if __name__ == "__main__":
    main()
