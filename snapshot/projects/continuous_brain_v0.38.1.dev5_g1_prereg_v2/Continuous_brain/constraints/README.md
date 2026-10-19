# Environment constraints

The CPU validation environment and DirectML environment are separate. Do not install this
project into the historical DirectML venv with an unconstrained `pip install`, because pip may
replace the torch build required by `torch-directml`.

Use `scripts/local/capture_directml_lock.py` inside a known-good DirectML environment first.
It writes exact installed versions of torch, torch-directml, numpy and Python provenance to a
constraints file. Recreate that environment with `pip install -c <lock> -e ".[directml]"`.
The repository does not guess a torch-directml compatibility pair.
