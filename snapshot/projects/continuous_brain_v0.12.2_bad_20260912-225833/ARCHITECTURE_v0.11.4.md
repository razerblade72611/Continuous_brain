# Protected growth architecture — v0.11.4 r2

The reported paired A100 experiment measured P4b completion of 38% at exact
expansion and 13% after fully visible v0.11.0 Phase A growth. Generalized
randomized-detour development reached 1%. These are user-provided prior results;
this release has not rerun that child's capability experiment.

## Residual visibility and learning

For normal inference, an existing residual stack computes
`output = input + gate * (grown - input)`. Gate zero returns the input directly;
gate one returns the grown output directly. Gate values are finite, bounded
checkpoint buffers, not learned parameters. Existing learned parameter counts
are unchanged. Migration copies every original tensor exactly.

The planner and option gates are scheduled at 0%, 2%, 5%, and 10%. During the
zero-gate stage, an explicit auxiliary forward exposes these two residuals to the
training objective; actual inference remains closed. The loss evaluates the
function whose residual weights it trains. There is no straight-through estimator
with a constant zero-gate loss. The auxiliary context is always restored and is
never serialized into a checkpoint. In later stages the training loss uses the
actual visible gate.

The target spatial and target visual pathways stay closed and frozen. General
non-planning growth remains at 1.0, making final bootstrap and world retention
necessary. Gate magnitudes bound interpolation coefficients, not residual norms
or behavioral changes; stage evaluations are the behavioral protection.

## Compatibility and rehearsal

The frozen teacher is reconstructed from the verified legacy tensors, with all
growth removed. Its parameter fingerprint must match the migration fingerprint.
No external teacher model is queried.

Planning and option task batches mix successful old Detour training sequences
with new randomized training sequences. Compatibility uses old Detour training
data separately. Option IDs are assigned by the frozen legacy selector, preserving
its established identities. Original k-means centers are absent from old
checkpoint metadata, so new cluster IDs cannot safely replace them. Termination
labels mark selector segment changes/endpoints; action/reward targets remain the
actual lived observations. The receipts record this label provenance.

Planner anchors compare latent state, demonstrated-action transitions, all-action
Q values, Q distributions, and strict teacher action ordering. Option anchors
compare selector probabilities, primitive policies, termination, and Q values/
ordering. Student option outputs receive the current student planner latent;
teacher outputs receive the frozen latent, covering their shared interface.

The actual architecture routes hierarchical options from planning latents, not
from a direct workspace-to-option connection. Workspace, visual, language,
motor, and value interfaces are protected separately by the general anchor on
every route. Only growth parameters for the selected route are trainable.

## State and evidence

Stages have cumulative fixed ceilings and prior-stage P4b admission proofs.
Checkpoint saves record dataset digests, settings, optimizer state, RNG state,
cumulative route counts, gates, and measured legacy tensor fingerprints. Resume
refuses protocol changes and loads protected model state strictly.

The Colab runner saves stage endpoints, streams logs to Drive, mirrors verified
periodic checkpoints every 250 updates, and reuses an evaluation only when its
checkpoint, protocol, source manifest, and runtime match. Existing endpoints are
never replaced by a later stage. Readiness requires complete measured evidence,
all four P4b passes, final retention, and ≥20% randomized development completion.

Development seeds are 41001–41100; reserved blind seeds 51001–51100 are not run
by the notebook. A software pass is not evidence that this training protocol
recovers or generalizes the mature policy.
