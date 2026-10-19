#!/usr/bin/env python3
import math
import warnings

import torch


class DMLSafeAdam:
    """Same Adam equations used by the DirectML qualification path."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)
        self.step_count = 0
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        self.step_count += 1

        bc1 = 1.0 - self.beta1 ** self.step_count
        bc2 = 1.0 - self.beta2 ** self.step_count
        sqrt_bc2 = math.sqrt(bc2)
        step_size = self.lr / bc1

        for p in self.params:
            g = p.grad
            if g is None:
                continue

            key = id(p)
            if key not in self.state:
                self.state[key] = {
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }

            m = self.state[key]["exp_avg"]
            v = self.state[key]["exp_avg_sq"]

            m.mul_(self.beta1)
            m.add_(g, alpha=1.0 - self.beta1)

            g2 = g * g
            v.mul_(self.beta2)
            v.add_(g2, alpha=1.0 - self.beta2)

            denom = v.sqrt()
            denom.div_(sqrt_bc2)
            denom.add_(self.eps)

            update = m / denom
            p.add_(update, alpha=-step_size)


def deterministic_grad(shape, step, index, dtype):
    n = 1
    for x in shape:
        n *= x
    base = torch.arange(1, n + 1, dtype=dtype).reshape(shape)
    sign = -1.0 if ((step + index) % 2) else 1.0
    return (
        sign * (0.011 * step) * torch.sin(base * (0.17 + 0.01 * index))
        + (0.003 * index) * torch.cos(base * 0.13)
    )


def run_dtype(dtype, steps=12):
    torch.manual_seed(73021)

    initial = [
        torch.randn(17, dtype=dtype),
        torch.randn(5, 7, dtype=dtype),
        torch.randn(3, 4, 2, dtype=dtype),
    ]

    reference = [
        torch.nn.Parameter(x.clone()) for x in initial
    ]
    candidate = [
        torch.nn.Parameter(x.clone()) for x in initial
    ]

    lr = 7e-4
    betas = (0.87, 0.993)
    eps = 3e-8

    real_adam = torch.optim.Adam(
        reference,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
    )
    safe_adam = DMLSafeAdam(
        candidate,
        lr=lr,
        betas=betas,
        eps=eps,
    )

    if dtype == torch.float64:
        rtol, atol = 1e-12, 1e-12
    else:
        rtol, atol = 2e-6, 2e-7

    print(f"dtype: {dtype}")
    print(f"steps: {steps}")

    for step in range(1, steps + 1):
        real_adam.zero_grad(set_to_none=True)
        safe_adam.zero_grad(set_to_none=True)

        for i, (rp, cp) in enumerate(zip(reference, candidate), start=1):
            g = deterministic_grad(tuple(rp.shape), step, i, dtype)
            rp.grad = g.clone()
            cp.grad = g.clone()

        real_adam.step()
        safe_adam.step()

        max_param_diff = 0.0
        max_m_diff = 0.0
        max_v_diff = 0.0

        for rp, cp in zip(reference, candidate):
            torch.testing.assert_close(
                cp,
                rp,
                rtol=rtol,
                atol=atol,
                msg=lambda msg: f"parameter mismatch at step {step}: {msg}",
            )

            real_state = real_adam.state[rp]
            safe_state = safe_adam.state[id(cp)]

            torch.testing.assert_close(
                safe_state["exp_avg"],
                real_state["exp_avg"],
                rtol=rtol,
                atol=atol,
                msg=lambda msg: f"exp_avg mismatch at step {step}: {msg}",
            )
            torch.testing.assert_close(
                safe_state["exp_avg_sq"],
                real_state["exp_avg_sq"],
                rtol=rtol,
                atol=atol,
                msg=lambda msg: f"exp_avg_sq mismatch at step {step}: {msg}",
            )

            max_param_diff = max(
                max_param_diff,
                float((cp - rp).abs().max().item()),
            )
            max_m_diff = max(
                max_m_diff,
                float(
                    (
                        safe_state["exp_avg"]
                        - real_state["exp_avg"]
                    ).abs().max().item()
                ),
            )
            max_v_diff = max(
                max_v_diff,
                float(
                    (
                        safe_state["exp_avg_sq"]
                        - real_state["exp_avg_sq"]
                    ).abs().max().item()
                ),
            )

        print(
            f"step={step:02d} "
            f"max_param_diff={max_param_diff:.3e} "
            f"max_m_diff={max_m_diff:.3e} "
            f"max_v_diff={max_v_diff:.3e}"
        )

    print(f"CPU_ADAM_EQUIVALENCE_OK: {dtype}")


def main():
    warnings.simplefilter("always")
    run_dtype(torch.float64)
    run_dtype(torch.float32)
    print("DML_SAFE_ADAM_CPU_VALIDATED")


if __name__ == "__main__":
    main()
