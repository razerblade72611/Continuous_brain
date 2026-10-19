# Continuous Brain — Parameter-Matched Ungated Recall Phase 1 v1

Status: TRAIN-only, pre-DEV.

This run is authorized only because the 49,280-parameter ungated recall control
passed DirectML/staged-gradient qualification.

## Frozen comparison

Control:
- same protected 14k checkpoint;
- same 36,992-parameter writer;
- same total additions as M2: 86,272;
- same recall parameter count as M2: 49,280;
- no recurrent-hidden/context input to recall;
- no contextual gate;
- no recall biases;
- zero-init output layer.

Training is exactly matched to M1.5/M2:
- 1600 TRAIN conditions, no replacement;
- schedule seed 73191;
- init seed 12345;
- LR 1e-4;
- DMLSafeAdam;
- exact LEFT/RIGHT pair each update;
- motor objective = 0.50 mean-step CE + 0.50 delayed-decision CE;
- world loss weight = 0.25;
- save every 100;
- whole-base verification every 50.

## Post-training gate

The runner immediately evaluates the fixed update-1600 control checkpoint on all
1600 TRAIN conditions using the same causal interventions previously used for
M1.5 and M2, then prints a three-way comparison:

M1.5 linear
vs.
parameter-matched ungated control
vs.
M2 gated

Primary interpretation:

- control ~= M2:
  M2's small advantage is explained by added memory-only capacity/nonlinearity;
  contextual gating is not established.

- M2 materially > control:
  evidence that hidden/context-dependent modulation adds value beyond matched
  memory-only capacity.

- control > M2:
  contextual gating is unnecessary or harmful for this task family.

DEV and TEST remain sealed throughout.
