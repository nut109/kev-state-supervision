"""Small data/metric helpers extracted from the original latent_run module."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def paired_interval(a, b, seed=0, replicates=2000):
    """World-clustered paired bootstrap, with seed results averaged per world."""
    x = defaultdict(list)
    for seed_id, (aa, bb) in enumerate(zip(a, b, strict=True)):
        if [row["id"] for row in aa] != [row["id"] for row in bb]:
            raise ValueError("paired predictions are not aligned")
        for row_a, row_b in zip(aa, bb, strict=True):
            x[row_a["world_id"]].append((seed_id, int(row_a["pred"] == row_a["label"])
                                             - int(row_b["pred"] == row_b["label"])))
    world_values = np.array([np.mean([value for _, value in entries])
                             for entries in x.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(world_values), size=(replicates, len(world_values)))
    boot = world_values[sample].mean(axis=1)
    per_seed = [np.mean([int(x["pred"] == x["label"]) - int(y["pred"] == y["label"])
                         for x, y in zip(aa, bb, strict=True)])
                for aa, bb in zip(a, b, strict=True)]
    return {"difference": float(world_values.mean()),
            "ci95": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
            "seed_differences": [float(v) for v in per_seed],
            "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else None,
            "worlds": len(world_values)}
