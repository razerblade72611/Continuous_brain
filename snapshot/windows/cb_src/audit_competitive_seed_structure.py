#!/usr/bin/env python3
import argparse
import collections
import json
import math
from pathlib import Path

from continuous_brain.cued_detour_world import CuedDetourWorld


ACTION_NAMES = {
    0: "idle",
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


def load(path):
    return json.loads(Path(path).read_text())


def canonical_scalar(v):
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return round(float(v), 8)
        return str(v)
    if isinstance(v, str):
        return v
    return None


def simple_value(v):
    s = canonical_scalar(v)
    if s is not None:
        return s

    if isinstance(v, (list, tuple)):
        if len(v) <= 12:
            vals = []
            for x in v:
                sx = canonical_scalar(x)
                if sx is None:
                    return None
                vals.append(sx)
            return vals

    # Handle small numpy-like vectors without importing numpy explicitly.
    if hasattr(v, "tolist"):
        try:
            x = v.tolist()
            return simple_value(x)
        except Exception:
            return None

    return None


def env_features(seed, corridor_length):
    env = CuedDetourWorld(
        seed=int(seed),
        corridor_length=float(corridor_length),
        max_episode_steps=120,
    )
    obs = env.reset()

    feat = {
        "seed": int(seed),
        "cue": str(getattr(env, "cue", "?")),
        "corridor_length": float(corridor_length),
    }

    agent = getattr(env, "agent", None)
    if agent is not None:
        try:
            feat["start_agent_x"] = round(float(agent[0]), 8)
            feat["start_agent_y"] = round(float(agent[1]), 8)
        except Exception:
            pass

    # Capture every simple public environment attribute so that a seed-driven
    # geometry variable cannot hide just because we did not know its name.
    for name, value in vars(env).items():
        if name.startswith("_"):
            continue
        if name in {"rng", "random", "np_random"}:
            continue
        sv = simple_value(value)
        if sv is not None:
            feat[f"env.{name}"] = sv

    # Also capture simple observation values at reset.
    if isinstance(obs, dict):
        for name, value in obs.items():
            sv = simple_value(value)
            if sv is not None:
                feat[f"obs.{name}"] = sv
    else:
        sv = simple_value(obs)
        if sv is not None:
            feat["obs"] = sv

    return feat


def row_map(result, key):
    return {
        int(r["seed"]): r
        for r in result["lengths"][key]["per_seed"]
    }


def fisher_two_sided(a, b, c, d):
    # Table:
    #             success fail
    # group A       a     b
    # group B       c     d
    n = a + b + c + d
    r1 = a + b
    c1 = a + c

    def hypergeom(x):
        return (
            math.comb(c1, x)
            * math.comb(n - c1, r1 - x)
            / math.comb(n, r1)
        )

    lo = max(0, r1 - (n - c1))
    hi = min(r1, c1)
    p_obs = hypergeom(a)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hypergeom(x)
        if px <= p_obs + 1e-15:
            p += px
    return min(1.0, p)


def fmt_counter(counter):
    return ", ".join(
        f"{k}:{v}"
        for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))
    )


def audit_length(key, results):
    print()
    print("=" * 88)
    print(f"CORRIDOR LENGTH {key}")
    print("=" * 88)

    maps = {name: row_map(res, key) for name, res in results.items()}
    seeds = sorted(maps["base"])
    for name, m in maps.items():
        if sorted(m) != seeds:
            raise RuntimeError(f"seed mismatch for {name} at {key}")

    print("\nClosed-loop success by cue:")
    for name in ("base", "uniform", "shared", "independent"):
        groups = collections.defaultdict(list)
        for seed in seeds:
            row = maps[name][seed]
            cue = row["junction"]["cue"]
            groups[cue].append(bool(row["closed_loop_success"]))

        parts = []
        for cue in sorted(groups):
            xs = groups[cue]
            parts.append(f"{cue}={sum(xs)}/{len(xs)}")
        print(f"  {name:>12}: " + "  ".join(parts))

    shared_success = {
        s for s in seeds if maps["shared"][s]["closed_loop_success"]
    }
    independent_success = {
        s for s in seeds if maps["independent"][s]["closed_loop_success"]
    }
    uniform_success = {
        s for s in seeds if maps["uniform"][s]["closed_loop_success"]
    }
    base_success = {
        s for s in seeds if maps["base"][s]["closed_loop_success"]
    }

    print("\nSuccess-set identity:")
    print("  shared successes:     ", len(shared_success))
    print("  independent successes:", len(independent_success))
    print("  intersection:         ", len(shared_success & independent_success))
    print("  shared-only:          ", sorted(shared_success - independent_success))
    print("  independent-only:     ", sorted(independent_success - shared_success))
    print("  uniform successes:    ", len(uniform_success))
    print("  base successes:       ", len(base_success))

    if shared_success == independent_success:
        print("  EXACT_MATCH_SHARED_INDEPENDENT: true")
    else:
        print("  EXACT_MATCH_SHARED_INDEPENDENT: false")

    if key == "0.300":
        print("\nSuccessful 0.3 seeds with cue:")
        for seed in sorted(shared_success | independent_success):
            cue = maps["shared"][seed]["junction"]["cue"]
            print(
                f"  seed={seed} cue={cue} "
                f"shared={int(seed in shared_success)} "
                f"independent={int(seed in independent_success)}"
            )

    print("\nJunction FULL argmax action distribution:")
    for name in ("base", "uniform", "shared", "independent"):
        counts = collections.Counter(
            ACTION_NAMES.get(
                int(maps[name][s]["junction"]["full_pred"]),
                str(maps[name][s]["junction"]["full_pred"]),
            )
            for s in seeds
        )
        print(f"  {name:>12}: {fmt_counter(counts)}")

    print("\nJunction SIDE prediction distribution:")
    for name in ("base", "uniform", "shared", "independent"):
        counts = collections.Counter(
            ACTION_NAMES.get(
                int(maps[name][s]["junction"]["side_pred"]),
                str(maps[name][s]["junction"]["side_pred"]),
            )
            for s in seeds
        )
        print(f"  {name:>12}: {fmt_counter(counts)}")

    # Reconstruct deterministic environment features from the same seeds.
    corridor_length = float(key)
    features = {
        s: env_features(s, corridor_length)
        for s in seeds
    }

    # Analyze cue explicitly using shared/independent common success set.
    common_success = shared_success & independent_success
    common_fail = set(seeds) - common_success
    cue_values = sorted({features[s]["cue"] for s in seeds})

    if len(cue_values) == 2:
        g0, g1 = cue_values
        a = sum(s in common_success and features[s]["cue"] == g0 for s in seeds)
        b = sum(s in common_fail and features[s]["cue"] == g0 for s in seeds)
        c = sum(s in common_success and features[s]["cue"] == g1 for s in seeds)
        d = sum(s in common_fail and features[s]["cue"] == g1 for s in seeds)
        print("\nCue association with common learned-condition success:")
        print(f"  {g0}: success={a} fail={b}")
        print(f"  {g1}: success={c} fail={d}")
        print(f"  Fisher two-sided p={fisher_two_sided(a,b,c,d):.12g}")

    # Find any reconstructed simple feature that partitions success strongly.
    all_keys = sorted(set().union(*(f.keys() for f in features.values())))
    skip = {"seed", "corridor_length"}
    candidates = []

    for feature in all_keys:
        if feature in skip:
            continue
        vals = [features[s].get(feature, "<missing>") for s in seeds]
        # Hash list values for grouping.
        vals = [
            tuple(v) if isinstance(v, list) else v
            for v in vals
        ]
        uniq = sorted(set(vals), key=str)
        if len(uniq) < 2 or len(uniq) > 10:
            continue

        groups = []
        for value in uniq:
            group_seeds = [
                s for s, v in zip(seeds, vals) if v == value
            ]
            succ = sum(s in common_success for s in group_seeds)
            groups.append((value, succ, len(group_seeds)))

        rates = [
            succ / n for _v, succ, n in groups if n
        ]
        spread = max(rates) - min(rates)
        if spread >= 0.25:
            candidates.append((spread, feature, groups))

    print("\nSimple reconstructed features with >=25-point success-rate spread:")
    if not candidates:
        print("  none")
    else:
        for spread, feature, groups in sorted(candidates, reverse=True)[:25]:
            desc = "; ".join(
                f"{value!r}:{succ}/{n}"
                for value, succ, n in groups
            )
            print(f"  {feature} spread={spread:.3f} -> {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir",
        default="state/competitive_decision_assays",
    )
    args = ap.parse_args()

    root = Path(args.dir)
    results = {
        name: load(root / f"{name}.json")
        for name in ("base", "uniform", "shared", "independent")
    }

    # Basic provenance consistency.
    sha = results["base"]["base_checkpoint_sha256"]
    for name, result in results.items():
        if result["base_checkpoint_sha256"] != sha:
            raise RuntimeError(f"base SHA mismatch for {name}")

    lengths = results["base"]["corridor_lengths"]
    for name, result in results.items():
        if result["corridor_lengths"] != lengths:
            raise RuntimeError(f"corridor length mismatch for {name}")
        if result["seed_base"] != results["base"]["seed_base"]:
            raise RuntimeError(f"seed base mismatch for {name}")
        if result["episodes_per_length"] != results["base"]["episodes_per_length"]:
            raise RuntimeError(f"episode count mismatch for {name}")

    print("base_checkpoint_sha256:", sha)
    print("seed_base:", results["base"]["seed_base"])
    print("episodes_per_length:", results["base"]["episodes_per_length"])

    for length in lengths:
        audit_length(f"{float(length):.3f}", results)

    print()
    print("SEED_STRUCTURE_AUDIT_COMPLETE")


if __name__ == "__main__":
    main()
