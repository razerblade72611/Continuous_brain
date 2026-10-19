# G0 Legacy Requalification — v0.38.1.dev0

## Purpose

G0 is the evidence and adapter gate **before composition**. It does not reopen any historical sealed TEST and it does not turn a module import into an admitted capability.

A candidate can advance only in this order:

1. identify immutable source root(s);
2. verify source, transitive support, evaluator, result, protocol, and learned-state hashes;
3. classify the historical lineage without erasing FAIL/diagnostic records;
4. write an adapter/state codec without modifying the historical code;
5. re-host the original metric and threshold semantics on a **fresh registered DEV family** where the historical evaluator is hard-coded to a spent TEST family;
6. only a candidate that reproduces its admission contract in isolation becomes eligible for G1–G3 composition.

A historical sealed TEST is never rerun to recreate evidence.

## Evidence source

The committed plan `legacy/g0_plan.json` was generated from David's read-only WSL evidence bundle:

- source bundle SHA-256: `76697543b19a2ecf7b6b2dfd45e0d84a7e0871ad8dc05368d53ab5f4a526e805`
- source repository commit: `c011d4048969b969520c6f8814f18e7ec11cc02e`
- candidate inventory SHA-256: `91c4c6df749971ab467e045c51f16cb668df15080f2c6d4fc3fb18b0c9b92b02`
- result files inventoried: 54
- checkpoint files inventoried: 520
- evaluator candidates inventoried: 134
- protected child checkpoint SHA-256: `086932388a9a410af46fea8e87cc63dd4b60161e03e247bf81fc88d28cfd577f`

The plan contains **hash expectations**, not copied historical code or checkpoints.

## Candidate dispositions

| Candidate | Historical record represented by plan | v0.38.1 disposition |
|---|---|---|
| `episodic_two_item_v023` | DEV + frozen TEST PASS | `READY_FOR_G0` |
| `episodic_goal_skill_v024` | integration PASS; exact upstream learned-state hashes recorded | `READY_FOR_G0` |
| `memory_guided_goal_options_v021` | mixed supporting pieces with retained temporal/termination failures | `SUPPORT_ONLY` |
| `consequence_replanning_v0252` | repaired line + admitted frozen TEST | `READY_FOR_G0` |
| `consequence_model_v0252` | learned state shared with admitted replanning line | `READY_FOR_G0_SHARED_STATE` |
| `discovery_embodiment_v026` | environment admission | `CONDITIONAL_EXTERNAL_ENV` |
| `persistent_self_v028` | DEV + frozen TEST PASS | `READY_FOR_G0` |
| `recurrent_self_trajectory_v031` | DEV + frozen TEST PASS | `READY_FOR_G0` |
| `autobiographical_discrepancy_v0322` | neutral-repair line + frozen TEST PASS | `READY_FOR_G0` |
| `event_consistent_self_development_v0341` | later lineage records PASS/closed; WSL evidence pins the recovered result/state files | `READY_FOR_G0` |
| `endogenous_goal_genesis_v0353` | v0.35.0 FAIL, v0.35.2 DEV PASS, v0.35.3 frozen TEST PASS described by later protocols; current candidate parse was stale | `NEEDS_REFRESHED_RESULT_PARSE` |
| `recursive_autobiographical_curriculum_v0361` | DEV + frozen TEST PASS | `READY_FOR_G0` |
| `working_memory_components_v013_v019` | mixed family containing admitted mechanisms and formal failures | `NEEDS_ADJUDICATION` |
| `self_broadcast_v029` | frozen TEST FAIL | `CONTROL_ONLY` |

The dispositions are intentionally asymmetric. A PASS family can be blocked by missing evidence; a FAIL family is never promoted because its module exists.

## Read-only verification

On David's WSL machine:

```bash
export CONTINUOUS_BRAIN_LEGACY_ROOT="$HOME/projects/continuous_brain_v0.12.2"
python scripts/local/g0_plan.py --legacy-root "$CONTINUOUS_BRAIN_LEGACY_ROOT"
```

This checks the committed hash plan against the named WSL root. It executes no historical test and writes nothing into the legacy tree.

To refresh the candidate parse after the v0.35.2/v0.35.3 spec expansion:

```bash
bash scripts/local/refresh_g0_inventory.sh "$CONTINUOUS_BRAIN_LEGACY_ROOT"
```

The refresh writes into this repository's `evidence/g0_runtime/` directory, never into the legacy root.

## What G0 is not

G0 does **not** mean:

- source file present;
- Python import succeeds;
- checkpoint deserializes;
- historical TEST is rerun;
- the mechanism works under composition;
- the mechanism is evidence for consciousness or sentience.

It is the prerequisite for testing those narrower functional questions safely.
