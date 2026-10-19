#!/usr/bin/env python3
"""Re-evaluate an I1 run directory (full report)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from continuous_brain_unified.integration.evaluate import evaluate, print_report  # noqa: E402
print_report(evaluate(Path(sys.argv[1])))
