# Parent Model Selection — 2026-09

Parents are defaults, not permanent architecture dependencies.

## Vision parent — DINOv3 ViT-B/16

`facebook/dinov3-vitb16-pretrain-lvd1689m`

Reason: strong universal dense visual representation at a practical size. It teaches the child's raw-pixel visual cortex through representation distillation. For low-resource experiments the old DINOv3 ViT-S lessons can still be imported; changing parent dimension does not change the child's latent bus.

## Language/reasoning parent — Qwen3.5-4B

`Qwen/Qwen3.5-4B`

Reason: strong current multimodal model, Apache-2.0, practical enough to run as a service on moderate hardware/cloud. For harder lessons use `Qwen/Qwen3.5-9B`; for occasional high-quality cloud teaching/evaluation use `Qwen/Qwen3.5-35B-A3B-FP8` or another future stronger parent.

The child language cortex is separate and randomly initialized. Qwen text is training evidence, not runtime identity.

## Temporal parent — V-JEPA 2 ViT-L

`facebook/vjepa2-vitl-fpc64-256`

Reason: approximately 0.3B-class video representation parent and much lighter than the 1B ViT-g options. It is used to teach temporal representations. The child's action-conditioned world model remains body-specific and trainable.

## Speech parent — Whisper large-v3-turbo

`openai/whisper-large-v3-turbo`

Reason: 809M ASR model with large-v3 lineage and a much smaller decoder. It can provide speech/transcript targets while a smaller child auditory cortex develops.

## Upgrade policy

Parent model names are stored in `parents.json`. Every replacement is an explicit developmental event. The project should periodically benchmark newly available models and upgrade only when teacher quality per unit compute improves materially.
