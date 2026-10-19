# Legacy binding contract

Set `CONTINUOUS_BRAIN_LEGACY_ROOT` to a root containing the historical `continuous_brain/` package and evaluator scripts.

The loader:

1. resolves every file under that root;
2. rejects path traversal;
3. checks SHA-256 before import;
4. detects an already-imported `continuous_brain` package from a different root and fails rather than shadowing it;
5. checks declared symbols;
6. reports missing or mismatched components as unbound.

A verified source binding is not a fresh scientific admission.
