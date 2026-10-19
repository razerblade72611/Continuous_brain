# Competence Roadmap — from teacher-assisted development to native competence

## What is realistic

A ~1M-parameter dialogue cortex cannot hold the broad language/world knowledge of a modern multi-billion-parameter teacher. The current cortex is a proof that teacher knowledge can be distilled into the persistent architecture, not a claim of teacher parity.

Teacher-level competence becomes a plausible engineering target only by scaling several things together:

1. **capacity** — add larger native semantic/language/reasoning expert networks rather than forcing all knowledge into the small executive
2. **curriculum/data** — millions to billions of grounded text/image/action/temporal lessons, not thousands
3. **world-model depth** — learn multi-step prediction and counterfactual rollouts
4. **episodic + semantic consolidation** — convert lived episodes into stable concepts while retaining provenance
5. **reasoning distillation** — train on solutions, critiques, counterfactuals and self-correction from stronger teachers
6. **continual learning without catastrophic forgetting** — replay, consolidation, adapters/experts and protected autobiographical memory
7. **active learning** — the individual asks teachers for help where uncertainty is high instead of labeling everything uniformly
8. **multi-teacher education** — visual, language, code, science and social teachers can supervise specialist cortical modules
9. **embodied curriculum** — real/simulated action consequences provide grounding unavailable from text-only distillation
10. **care/value integration** — social consequence prediction must affect executive choice, not remain only a conversational head

## Unique scaling strategy

Do not turn the whole system into one giant transformer. Keep a persistent recurrent individual at the center and add specialist neural cortices around it:

`perception experts -> shared persistent workspace <-> world model / memory / language experts -> hierarchical action`

Teachers may remain frozen during early education, then be replaced by distilled native experts where independence matters. A mature system may still retain optional external tools, just as a person uses books/calculators, without those tools becoming the individual's identity.

## Success criterion

The target is not merely teacher imitation. It is a system that can:

- learn teacher-level abstractions
- ground them in its own perception/actions
- preserve a continuous autobiographical history
- reason from its own state and memories
- recognize uncertainty
- improve through later experience
- develop cooperative/caring behavior through consequence-aware value learning

Teacher parity should therefore be measured separately by domain: language, reasoning, vision, memory, embodied planning, continual adaptation and social decision quality.
