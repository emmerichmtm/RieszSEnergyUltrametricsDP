from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Dict, List

from ultrametric_dp import (
    brute_force_best,
    dp_with_choices,
    primate_example_tree,
    random_binary_ultrametric,
    reconstruct_subset,
)


def run_validation() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []

    # Hand-checked named example.
    root = primate_example_tree()
    for s in (0.5, 1.0, 2.0, 3.0):
        for k in (2, 3, 4):
            table, choice = dp_with_choices(root, s)
            dp_val = table[root][k]
            dp_subset = reconstruct_subset(root, k, choice)
            bf_val, bf_subset = brute_force_best(root, k, s)
            rows.append(
                {
                    "family": "primate",
                    "seed": None,
                    "n": 4,
                    "k": k,
                    "s": s,
                    "dp_value": dp_val,
                    "dp_subset": list(dp_subset),
                    "bruteforce_value": bf_val,
                    "bruteforce_subset": list(bf_subset),
                    "abs_diff": abs(dp_val - bf_val),
                    "ok": abs(dp_val - bf_val) <= 1e-12,
                }
            )

    # Deterministic random batch.
    seeds = list(range(20))
    sizes = [5, 6, 7, 8, 9, 10]
    exponents = [0.5, 1.0, 2.0, 3.0]
    for seed in seeds:
        for n in sizes:
            root = random_binary_ultrametric(n, 1000 + 37 * seed + n)
            for s in exponents:
                k = max(2, n // 2)
                table, choice = dp_with_choices(root, s)
                dp_val = table[root][k]
                dp_subset = reconstruct_subset(root, k, choice)
                bf_val, bf_subset = brute_force_best(root, k, s)
                rows.append(
                    {
                        "family": "random",
                        "seed": seed,
                        "n": n,
                        "k": k,
                        "s": s,
                        "dp_value": dp_val,
                        "dp_subset": list(dp_subset),
                        "bruteforce_value": bf_val,
                        "bruteforce_subset": list(bf_subset),
                        "abs_diff": abs(dp_val - bf_val),
                        "ok": abs(dp_val - bf_val) <= 1e-12,
                    }
                )

    max_abs_diff = max(row["abs_diff"] for row in rows) if rows else 0.0
    num_ok = sum(1 for row in rows if row["ok"])

    by_n: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        by_n[int(row["n"])].append(float(row["abs_diff"]))

    summary = {
        "num_instances": len(rows),
        "num_ok": num_ok,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean(float(row["abs_diff"]) for row in rows) if rows else 0.0,
        "max_abs_diff_by_n": {str(n): max(vals) for n, vals in sorted(by_n.items())},
        "instances": rows,
    }
    return summary


if __name__ == "__main__":
    data = run_validation()
    with open("validation_results.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(json.dumps({k: data[k] for k in ("num_instances", "num_ok", "max_abs_diff", "mean_abs_diff")}, indent=2))
