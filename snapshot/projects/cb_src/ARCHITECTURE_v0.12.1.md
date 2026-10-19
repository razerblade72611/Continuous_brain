# Continuous Brain v0.12.1 — unified workspace integration

v0.12.1 is an integration and backend-hardening release. It retains the v0.12.0 lifelong runtime and the complete v0.11.4 legacy lineage, bundles the qualified 14,000-update InfantBrain checkpoint, and integrates the competitive-broadcast research branch around that frozen child.

## Two preserved model lineages

The v0.12 `DevelopmentBrain` remains the persistent local lifetime experiment. Its model/schema/build are unchanged (`v120-local-development-r1`) so existing `~/ContinuousBrain/individual_v120` state remains valid.

The bundled high-capacity checkpoint remains an `InfantBrain` checkpoint. It is not converted into the small v0.12 model. Its SHA-256 is:

`086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

It records 14,000 updates, `balanced-v110` growth, and 384/768 vision-parent heads.

## Competitive broadcast v3

`CompetitiveInfantBrain` freezes the entire learned base and pins it to eval mode. Zero-initialized residual adapters allow an optional broadcast to influence recurrent hidden state, motor workspace and world-model workspace. Disabling competition routes through the exact original child.

Shared selection uses one selector for all three consumers. Independent selection uses three selectors. Uniform selection holds the shared query at zero while residual capacity can still learn.

v0.12.1 exposes `CorticalNucleus.project_streams()` and `forward_from_streams()` so the frozen recurrent core and competitive selector reuse the exact same four projected streams. This adds no parameters and changes no state-dict keys.

## DirectML hardening

Two observed or strongly localized backend seams are addressed without changing CPU/CUDA behavior:

- `one_hot_action()` uses a scatter-free equality construction only on DirectML/PrivateUse1; CPU/CUDA retain `F.one_hot`.
- the DirectML-safe GRU decomposition uses explicit gate slicing instead of `Tensor.chunk`. CPU/CUDA still use native `nn.GRUCell` through `DMLSafeGRUCell.forward`.

No hidden state is detached. No truncated BPTT is enabled. Temporal credit assignment remains part of the competitive experiment.

## Qualification boundary

CPU validation can establish mathematical equivalence, checkpoint compatibility and preservation, but it cannot establish DirectML correctness on the RX 6600. `scripts/qualify_directml_v121.sh` therefore runs each DirectML gate in a fresh process: tensor/one-hot, recurrent chains of 1/2/5/10/20 steps, the real frozen child core, a full competitive one-step motor backward, and one complete 0.3-corridor training episode.

A failed gate stops qualification. The release does not silently fall back to CPU.
