import argparse
import torch
import torch_directml
from torch.nn import functional as F

from continuous_brain.evaluate_competitive_infant import _load_base_checkpoint
from continuous_brain.competitive_infant_brain import CompetitiveInfantBrain

CHECKPOINT = "state/child_v113_dml_smoke/infant_training.pt"
SNAPSHOT = "/tmp/cb_v122_real_snapshot.pt"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--case",
        required=True,
        choices=[
            "zero",
            "recurrent_only",
            "motor_only",
            "both_shared_leaf",
            "both_separate_leaves",
        ],
    )
    args = p.parse_args()

    dml = torch_directml.device()

    snap = torch.load(SNAPSHOT, map_location="cpu", weights_only=True)
    base, cfg, _, sha = _load_base_checkpoint(CHECKPOINT)
    assert sha == snap["checkpoint_sha256"]

    model = CompetitiveInfantBrain.grow_from(base, use_shared=True).to(dml)
    model.train(True)

    with torch.no_grad():
        model.recurrent_residual.linear.weight.zero_()
        model.motor_residual.linear.weight.zero_()
        if args.case in ("recurrent_only", "both_shared_leaf", "both_separate_leaves"):
            model.recurrent_residual.linear.weight.fill_(1e-4)
        if args.case in ("motor_only", "both_shared_leaf", "both_separate_leaves"):
            model.motor_residual.linear.weight.fill_(1e-4)

    base_hidden = snap["base_hidden"].to(dml)

    # Use the exact zero-init selector broadcast value numerically, but make it
    # a leaf so the selector itself is absent from this backward graph.
    broadcast_cpu = torch.stack(snap["streams"], dim=1).mean(dim=1)

    if args.case == "both_separate_leaves":
        recurrent_broadcast = broadcast_cpu.to(dml).clone().requires_grad_(True)
        motor_broadcast = broadcast_cpu.to(dml).clone().requires_grad_(True)
    else:
        shared = broadcast_cpu.to(dml).clone().requires_grad_(True)
        recurrent_broadcast = shared
        motor_broadcast = shared

    competitive_hidden = base_hidden + model.recurrent_residual(recurrent_broadcast)
    workspace = model._workspace_from_hidden(competitive_hidden)
    motor_workspace = workspace + model.motor_residual(motor_broadcast)
    logits = model.base.motor(motor_workspace)

    target = torch.tensor([snap["target"]], dtype=torch.long, device=dml)
    loss = F.cross_entropy(logits, target)

    print("case:", args.case)
    print("loss:", float(loss.detach().cpu()))
    print("starting backward...")
    loss.backward()

    rg = model.recurrent_residual.linear.weight.grad
    mg = model.motor_residual.linear.weight.grad

    print("recurrent residual grad max:", None if rg is None else float(rg.abs().max().cpu()))
    print("motor residual grad max:", None if mg is None else float(mg.abs().max().cpu()))

    if args.case == "both_separate_leaves":
        print("recurrent broadcast grad max:", float(recurrent_broadcast.grad.abs().max().cpu()))
        print("motor broadcast grad max:", float(motor_broadcast.grad.abs().max().cpu()))
    else:
        print("shared broadcast grad max:", float(recurrent_broadcast.grad.abs().max().cpu()))

    print("CASE_PASS:", args.case)


if __name__ == "__main__":
    main()
