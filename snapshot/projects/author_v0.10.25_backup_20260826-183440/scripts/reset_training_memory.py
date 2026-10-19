from __future__ import annotations

import argparse
from pathlib import Path

from agent_core.learning import reset_learning_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Clear v0.8 persistent learning/memory state only.")
    parser.add_argument("--state-dir", default="state")
    parser.add_argument("--yes", action="store_true", help="Required acknowledgement for standalone reset.")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing to clear state without --yes")
    resolved = reset_learning_state(Path(args.state_dir))
    print(f"Cleared persistent learning/memory state: {resolved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
