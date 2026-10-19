import argparse
import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import _load_base_checkpoint
from continuous_brain.competitive_infant_brain import ZeroInitResidual

CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
SNAPSHOT = "/tmp/cb_v122_real_snapshot.pt"

CASES = (
    "all_trainable",
    "all_frozen",
    "motor_trainable",
    "workspace_trainable",
    "workspace_motor_trainable",
)


def set_requires_grad(module, value: bool):
    for p in module.parameters():
        p.requires_grad_(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    args = parser.parse_args()

    dml = torch_directml.device()
    snap = torch.load(SNAPSHOT, map_location="cpu", weights_only=True)

    base, cfg, _, sha = _load_base_checkpoint(CHECKPOINT)
    assert sha == snap["checkpoint_sha256"]
    base.eval()

    # Control ONLY whether the frozen base requests parameter gradients.
    if args.case == "all_trainable":
        # _load_base_checkpoint returns the historical child with ordinary
        # trainable base parameters (apart from intentionally frozen targets).
        pass
    else:
        set_requires_grad(base, False)
        if args.case in ("motor_trainable", "workspace_motor_trainable"):
            set_requires_grad(base.motor, True)
        if args.case in ("workspace_trainable", "workspace_motor_trainable"):
            set_requires_grad(base.nucleus.workspace, True)

    base = base.to(dml)

    # Standalone adapters exactly match the earlier passing leaf-broadcast
    # diagnostic. They remain zero-initialized in every case.
    recurrent_residual = ZeroInitResidual(cfg.bus_dim, cfg.nucleus_dim).to(dml)
    motor_residual = ZeroInitResidual(cfg.bus_dim, cfg.bus_dim).to(dml)

    base_hidden = snap["base_hidden"].to(dml)
    broadcast_cpu = torch.stack(snap["streams"], dim=1).mean(dim=1)
    broadcast = broadcast_cpu.to(dml).clone().requires_grad_(True)

    competitive_hidden = base_hidden + recurrent_residual(broadcast)
    workspace = base.nucleus.growth(base.nucleus.workspace(competitive_hidden))
    motor_workspace = workspace + motor_residual(broadcast)
    logits = base.motor(motor_workspace)

    target = torch.tensor([snap["target"]], dtype=torch.long, device=dml)
    loss = F.cross_entropy(logits, target)

    total_base = sum(p.numel() for p in base.parameters())
    trainable_base = sum(p.numel() for p in base.parameters() if p.requires_grad)
    workspace_trainable = sum(p.numel() for p in base.nucleus.workspace.parameters() if p.requires_grad)
    motor_trainable = sum(p.numel() for p in base.motor.parameters() if p.requires_grad)

    print("case:", args.case)
    print("checkpoint:", sha)
    print("base params trainable/total:", trainable_base, "/", total_base)
    print("workspace trainable params:", workspace_trainable)
    print("motor trainable params:", motor_trainable)
    print("loss:", float(loss.detach().cpu()))
    print("starting backward...")

    loss.backward()

    print("broadcast grad max:", float(broadcast.grad.abs().max().cpu()))
    print("recurrent residual grad max:", float(recurrent_residual.linear.weight.grad.abs().max().cpu()))
    print("motor residual grad max:", float(motor_residual.linear.weight.grad.abs().max().cpu()))

    # We do not optimize the base in this diagnostic.  Base grads exist only to
    # test whether DirectML behaves differently when parameter gradients are
    # requested through the same forward math.
    print("CASE_PASS:", args.case)


if __name__ == "__main__":
    main()
