# Working Memory Shuffled-Control Mechanism Diagnostic v1

This diagnostic responds to the above-chance shuffled-label control in the
frozen-writer representation-reuse assay.

Important correction:
the original shuffled control already permuted individual examples, not whole
structural-condition pairs.

The goal here is not to estimate a more precise permutation p-value. The
above-chance result is already large enough to demand explanation.

We compare three randomization mechanisms:

1. `example_permutation`
   - reproduces the original control.

2. `pair_orientation`
   - one random normal/inverted pseudo-label orientation per structural
     condition;
   - pair labels remain complementary.

3. `pair_orientation_exact_balanced`
   - same as #2, but exactly half of conditions are normal and half inverted;
   - forces global pseudo-label agreement with true cue identity to 50%.

For each trained shuffled reader we measure:
- pseudo-label / true-label agreement in training targets;
- fit to the pseudo-labels;
- true-label train and test accuracy;
- cosine with the real-label reader decision vector;
- cosine with the empirical cue axis `mean(RIGHT memory - LEFT memory)`.

A cue-axis alignment mechanism predicts that shuffled true-label accuracy should
track cosine alignment with the real/cue directions. If the exactly-balanced
pair shuffle removes the effect, accidental true-cue agreement is implicated.
If it remains strongly above chance without cue-axis alignment, another leak or
optimization artifact must be found before DEV.

TRAIN only. DEV and TEST remain sealed.
