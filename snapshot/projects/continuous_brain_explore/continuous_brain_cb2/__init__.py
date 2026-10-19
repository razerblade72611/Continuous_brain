"""Continuous Brain 2, scale-first kernel (CB2-S1). Exploration lane, ENGINEERING_ONLY.

Four contracts are fixed here and everything behind them is replaceable:

- ``archive``: the experience archive (Parquet shards, two-axis provenance, manifests, a hash chain over the lived
  stream);
- ``world_model``: the predictive core behind ``step`` / ``scan`` / ``imagine``;
- ``organs``: the organ interface;
- ``consolidation``: the one recorded event that changes what the individual has learned.

See ``docs/explore/CB2_S1.md``.
"""
