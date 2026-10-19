#!/usr/bin/env bash
set -euo pipefail
continuous-brain --state-dir state/my_brain --visual --open-browser "$@"
