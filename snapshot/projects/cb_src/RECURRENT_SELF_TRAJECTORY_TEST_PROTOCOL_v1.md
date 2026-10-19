# Continuous Brain v0.31.1 — Frozen Recurrent Self-Trajectory TEST

## Authorization

v0.31.0 passed every DEV gate.

Key DEV results:
- teacher-forced world MAE: 0.00774
- teacher-forced self MAE: 0.00736
- teacher-forced primary intention: 0.99536
- teacher-forced mode: 0.99854
- teacher-forced active goal: 0.99390
- agency: 1.00000

Eight-step autonomous rollout:
- world MAE: 0.01878
- self MAE: 0.01886
- primary intention: 0.98291
- mode: 0.99854
- active goal: 0.98145

Causal:
- explicit intention trajectory choice: 1.00
- scalar-only reconstruction: 0.50
- donor-primary trajectory flip: 1.00
- surprise preserves primary and enters INVESTIGATE: 1.00
- primary resumes after investigate resolves: 1.00

Trajectory TEST remained sealed.

## Frozen TEST

No training.

Use:
- exact v0.31.0 recurrent self-trajectory model
- trajectory TEST IDs 20000..21023

All 1024 TEST trajectories are used for both:
- teacher-forced evaluation
- autonomous eight-step rollout

After this run the trajectory TEST is spent regardless of outcome.

## Frozen thresholds

Exactly preserve v0.31.0 thresholds.

Teacher-forced:
- world MAE <= .030
- self MAE <= .025
- primary >= .985
- mode >= .985
- agency >= .990

Autoregressive horizon 8:
- world MAE <= .075
- self MAE <= .060
- primary >= .950
- mode >= .950
- active goal >= .950

## Fresh explicit-intention intervention

256 new contexts are generated.

Within every A/B pair:
- external world is identical
- unresolved A == unresolved B
- confidence is identical
- uncertainty is identical
- prediction error is identical
- agency is identical

Only explicit primary intention differs.

Candidate trajectories:
- WORK_A x3
- WORK_B x3

Required:
- explicit intention selects its matching trajectory >= .95
- scalar-only tied reconstruction <= .60
- donor-primary swap flips trajectory >= .95

The scalar-only control is deliberately incapable of recovering whether an exact
tie previously belonged to A or B; its fixed tie break chooses A.

## Fresh interruption/resumption intervention

128 new identities and starting states.

Start:
- primary A
- NORMAL

Unexpected failed WORK_A:
- actual self diverges from ordinary predicted/expected self
- primary must remain A
- mode must become INVESTIGATE

Then repeated OBSERVE actions reduce uncertainty/error.

Required:
- primary A survives and INVESTIGATE begins = 1.00
- primary A resumes when mode returns NORMAL = 1.00
- ordinary-vs-surprise self divergence >= .10
- model-predicted-vs-surprise self divergence >= .10

## Preservation

Hash before/after:
- v0.31 DEV result
- v0.31 recurrent trajectory model
- historical InfantBrain
- admitted working memory
- episodic reader
- persistent self artifacts
- v0.29 TEST
- v0.30 result/repair/diagnostic

No protected artifact may change.

The unused v0.30 TEST remains untouched.

## Meaning of PASS

A PASS supports this functional claim on held-out trajectories:

The system can recurrently imagine eight-step future self/world trajectories
while preserving a discrete continuing primary intention separately from a
temporary uncertainty-management mode.

Changing only the continuing primary intention changes the selected imagined
personal trajectory even when both external world and continuous self variables
are identical.

Unexpected events can temporarily recruit information seeking without erasing
the underlying commitment, which later resumes.

This remains a computational and functional result. It is not sufficient
evidence of subjective experience, phenomenal consciousness, sentience, qualia,
emotion or moral status.

## Stop rule

After this run:
- trajectory TEST 20000..21023 is spent
- never tune v0.31 against it

If PASS:
- freeze v0.31 as the recurrent self-trajectory milestone
- next milestone should connect imagined personal futures to autobiography:
  store the intended self trajectory before acting, later compare it with the
  actual autobiographical trajectory, and let accumulated self-discrepancies
  change future commitment formation

If FAIL:
- preserve formal TEST FAIL
- do not repair against TEST
- move the architectural lesson into a new development family
