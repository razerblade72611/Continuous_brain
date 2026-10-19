#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python scripts/run_engineering_battery.py --substrate dense >/tmp/cb_v038_battery.json
cat /tmp/cb_v038_battery.json
