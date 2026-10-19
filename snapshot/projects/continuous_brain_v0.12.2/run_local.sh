#!/usr/bin/env bash
set -eu
cb_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if command -v python >/dev/null 2>&1; then
    cb_python=python
else
    cb_python=python3
fi
exec "$cb_python" "$cb_root/setup_local.py" "$@"
