# Continuous Brain — Working Memory Recall Comparison v2

Status: **qualification correction before production training**

No production M1.5 or M2 training has begun. DEV and TEST remain sealed.

## Why v2 exists

Recall Qualification v1 correctly established:
- exact zero-init no-op;
- staged memory-dependent recall gradients;
- bit-exact protected base;
- DirectML viability.

It also exposed a comparison confound:

- M1.5 direct recall uses `P(m)`;
- M2 v1 used `sigmoid(G[h,m]) * P(m)`;
- with zero gate parameters, M2's initial multiplier was `0.5`.

The real update-2 projection gradients showed the consequence directly:

- M1.5: approximately `0.00149961`
- M2 v1: approximately `0.000749805`

Thus M2 v1 had a 2x lower effective initial projection learning rate. That is
unacceptable for a controlled M1.5-vs-M2 behavioral comparison.

## Corrected normalized gate

M2 v2 is:

    decoded_t     = P_0(m_t)
    gate_t        = sigmoid(G([h_t,m_t]))
    multiplier_t  = 2 * gate_t
    residual_t    = multiplier_t * decoded_t

At zero gate parameters:

    gate_t = 0.5
    multiplier_t = 1.0

Therefore M2's direct recall path is initially equivalent in gain to M1.5:

    M1.5 residual = P_0(m_t)
    M2 v2 residual = 1.0 * P_0(m_t)

The projection remains bias-free and zero-initialized, so both remain exact
historical no-ops at construction.

## Stronger cross-arm control

Because the M2 gate cannot receive task gradient until decoded memory is
nonzero, M1.5 and M2 v2 should match through the first three qualification
forwards in:

- task losses;
- writer memory magnitude;
- writer candidate/write-gate gradients;
- recall residual magnitude;
- recall projection gradients.

M2 is allowed to differ only by acquiring nonzero gate gradients on update 3,
after that forward has completed.

`compare_recall_qualification_v2.py` enforces this cross-arm equivalence.

## Interpretation remains unchanged

- M1.5 ~= M2 > M1:
  direct workspace access/gain is sufficient; prefer M1.5.
- M2 > M1.5:
  contextual recall is promising, but gating-specific credit still requires a
  later parameter-matched ungated control.
- neither > M1:
  direct recall is insufficient; return to storage/integration.
- M1.5 > M2:
  contextual gating is unnecessary or harmful.

No 25-update production smoke or 1600-update arm training is authorized until
Recall Qualification v2 passes both arms and the cross-arm equivalence gate.
