# v0.9.4 release validation

Validated in the build environment:

- `pytest -q tests/test_v090.py tests/test_v092.py tests/test_v093.py` -> 11 passed
- `pytest -q tests/test_v094.py` -> 4 passed
- simulated v0.9.3 3,000-update checkpoint migration -> passed
  - continued at update 3001
  - optimizer safely reinitialized
  - new dorsal/EMA/context modules detected
  - old child checkpoint loaded without unrelated-state loss
- synthetic dual-stream dynamics learning smoke -> passed as a learning sanity check
  - dorsal next-state error improved over no-change baseline
  - structured next-state error improved over no-change baseline
  - both action-advantage metrics became positive

The synthetic smoke is not evidence that the user's real child has passed P3. The real held-out foraging evaluation must decide that.
