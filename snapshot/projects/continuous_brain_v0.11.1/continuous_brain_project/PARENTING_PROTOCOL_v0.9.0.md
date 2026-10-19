# Parenting Protocol v0.9.0

## Principle

Parents teach; they do not become the child.

A parent may supply:
- embeddings or representation targets
- demonstrations
- language responses and explanations
- curriculum examples
- corrections
- safety/value examples
- evaluations

A parent does not own:
- the child's recurrent state
- autobiographical memory
- identity
- goals
- value weights
- motor weights
- world-model weights
- child cortical weights

## Lifecycle

1. **Dependence** — parent authority near 1.0; extensive demonstration and correction.
2. **Imitation** — child learns teacher targets on held-out observations.
3. **Joint control** — child acts increasingly often while parent remains available.
4. **Supervised independence** — parent observes/evaluates but rarely intervenes.
5. **Graduation** — parent is absent for that capability; teacher-free tests must pass.
6. **Continual development** — parents can be temporarily reintroduced for new domains.

## Automatic weaning gate

The registry stores separate authority per capability. By default three consecutive held-out passes permit one 0.20 authority reduction. A failure resets the pass streak. The policy is deliberately conservative and inspectable.

## Parent upgrade

A better future model is registered as a new parent ID, not silently overwritten. The previous parent remains in history. New teacher targets can use a new detachable projection head while the child's latent bus remains unchanged.

Example:

```bash
python -m continuous_brain.parented_cli upgrade-parent \
  --state-dir state/child_v090 \
  --capability language \
  --parent-id language.qwen.future \
  --model FutureOrg/BetterModel \
  --interface openai-chat \
  --endpoint http://127.0.0.1:9000/v1/chat/completions
```

## Anti-shortcut rules

- Parent outputs carry provenance.
- Qualification runs child-only.
- Teacher calls are not permitted inside the child checkpoint.
- A teacher response is not an autobiographical memory of a lived event.
- Parent-specific heads can be removed after graduation.
- Value/care behavior is trained into the child's value/executive path, not asserted by a system prompt.
