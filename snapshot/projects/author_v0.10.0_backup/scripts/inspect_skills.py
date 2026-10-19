#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_core.embodied_skills import EmbodiedSkillLibrary
from agent_core.learning import LearningPaths


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect AUTHOR per-agent embodied skill checkpoints.")
    parser.add_argument("--state-dir", default="state")
    args = parser.parse_args()
    paths = LearningPaths.from_root(Path(args.state_dir)).ensure()
    payload = {}
    for label in ("A", "B"):
        lib = EmbodiedSkillLibrary(paths.agent_skill_checkpoint(label), agent_label=label)
        payload[label] = lib.compact_summary()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
