# Continuous Brain v0.7.1 — Grounded Dialogue Architecture

v0.7.1 keeps the persistent individual architecture intact and upgrades only the learned communication layer.

## Core individual

1. **Persistent self + lifetime store** — one individual ID, state, autobiographical experiences, lineage, goals and resumable checkpoints.
2. **Recurrent executive/workspace (~32.1M parameters in the standard v0.6/v0.7 lineage)** — integrates sensed state, retrieved memory, goals, skills and prediction signals over time.
3. **World/prediction learning** — predicts consequences and supplies learning/surprise signals; future imagination must remain provenance-labelled as imagined rather than lived.
4. **Hierarchical skills + learned motor cortex** — executive selects skills; a separately trained motor network realizes appropriate low-level movement where enabled.
5. **Vision** — frozen DINOv3 extracts pretrained visual representations; a small learned adapter maps them into the individual's own semantic space.
6. **Semantic/dialogue cortex (~1.07M parameters in v0.7.1 defaults)** — byte-level neural question encoder/answer decoder plus visual concept, care/social and dialogue-intent heads.
7. **Grounding router** — identity comes from `SelfModel.name`, current-goal reports from the persistent intention, and temporal reports from the latest real lifetime experience. Unknown questions are rejected rather than mapped to an unrelated learned answer.
8. **External teachers are developmental scaffolding** — Qwen/VLM teachers create supervision during training only. They do not own memory, choose actions, or generate qualified runtime conversation.

## v0.7.1 dialogue intents

`vision`, `state`, `reasoning`, `care`, `identity`, `temporal`, `unknown`.

The new neural intent head lets the system distinguish a known question family from an out-of-distribution question. Runtime uses a confidence threshold; low-confidence or `unknown` questions produce the explicit response:

> I don't understand that question yet.

This is deliberately more conservative than fluent guessing.

## Temporal compatibility

The semantic nursery now uses the same 8-value recent-experience tail used by the persistent lifetime memory:

- six-way learned skill one-hot
- clipped reward
- success flag

That gives the dialogue cortex a training path for questions such as “What did you do just before this?” and “Did your last action succeed?” without inventing a parallel memory representation.
