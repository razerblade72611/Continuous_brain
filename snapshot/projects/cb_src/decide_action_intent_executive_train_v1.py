#!/usr/bin/env python3
"""Exit 0: full TRAIN mapping pass; 4: valid failure; 2: invalid evidence."""
import argparse
import json
from pathlib import Path
import sys

from continuous_brain.action_intent_gate_v1 import decide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = decide(json.loads(Path(args.input).read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        result = {'format': 'action_intent_train_decision_v1_1', 'pass': False,
                  'evidence_valid': False, 'error': str(exc)}
        code = 2
    else:
        result['evidence_valid'] = True
        code = 0 if result['pass'] else 4
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    for key, passed in result.get('checks', {}).items():
        print(f"{key}: {'PASS' if passed else 'FAIL'}")
    if code == 2:
        print('INVALID_EVALUATION_EVIDENCE:', result['error'])
    print('ACTION_INTENT_TRAIN_GATE:', 'PASS' if result['pass'] else 'FAIL')
    print('SCOPE: teacher-prefix response mapping; autonomous control evaluated separately')
    sys.exit(code)


if __name__ == '__main__':
    main()
