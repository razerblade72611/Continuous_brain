# Continuous Brain v0.3.0

## Added
- Pluggable visual foundation-model interface.
- Frozen Meta DINOv3 ViT-S/16 provider through Transformers.
- Dependency-free `pixel-stats` provider for pipeline testing.
- Small trainable residual visual adapter with learnable gate.
- Visual episodic memory in the existing SQLite database.
- Curiosity signal from prediction surprise + visual novelty.
- Homeostatic energy-deficit drive.
- Dashboard metrics for vision, novelty, curiosity, drive, gate and visual memory.
- v0.2 checkpoint migration with compatible-weight preservation.

## Preserved
- v0.2 structured-state perception dimensions.
- GRU core dimensions and names.
- Q/world/reward heads.
- Existing episodic memories.
- Optional Ollama planner.
- Browser visualizer and pause/speed controls.

## Intentional limitation
DINOv3 is an image representation model, not yet an action-conditioned video world model. v0.4 is intended to introduce the provider/cache/distillation interfaces needed for V-JEPA-style temporal prediction and cloud-GPU teacher/student training.
