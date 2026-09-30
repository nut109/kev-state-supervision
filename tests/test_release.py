"""Release checks for the portable data, model, and result index."""
import csv
import json
import math
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F

from kev.latent_data import generate_causal_splits, solve_rules
from kev.latent_iit import HardRuleMachine
from kev.latent_iit_run import pack_rules
from kev.latent_iit_trials import state_loss


def test_deterministic_world_splits_and_trace():
    sizes = {"train": 40, "val": 20, "stress_val": 20, "id_test": 20, "ood_test": 20}
    a = generate_causal_splits(91, sizes)
    b = generate_causal_splits(91, sizes)
    assert a == b
    worlds = [{row["world_id"] for row in rows} for rows in a.values()]
    assert sum(map(len, worlds)) == len(set.union(*worlds))
    row = next(row for row in a["train"] if row["family"] == "rule")
    labels = pack_rules([row])
    profile = next(item for item in row["rule_inputs"] if item["subject"] == row["subject"])
    trace = [int(profile["initial"])] + [int(value) for value in solve_rules(
        profile["initial"], row["path"], profile["gates"])]
    assert labels["trajectories"][0] == trace
    assert row["options"][row["label"]] == ("Yes" if trace[-1] else "No")


def test_hard_rollout_and_process_loss_backward():
    rows = [row for row in generate_causal_splits(29, {"train": 40, "val": 20,
            "stress_val": 20, "id_test": 20, "ood_test": 20})["train"]
            if row["family"] == "rule"][:3]
    target = pack_rules(rows)
    model = HardRuleMachine(hidden=64, max_depth=10)
    initial = model.gold_logits(target["initial"], 2)
    ops = model.gold_logits(target["operators"], 4)
    gates = model.gold_logits(target["gates"], 3)
    answer, states, raw = model.rollout(initial, ops, gates, target["depths"],
                                        target["options"], loops=3, return_step_logits=True)
    assert answer.shape == (3, 2) and states.shape == (3, 4, 2)
    loss = F.cross_entropy(answer, target["labels"]) + state_loss({
        "parser": (initial, ops, gates), "step_logits": raw,
        "depths": target["depths"]}, rows)
    loss.backward()
    assert torch.isfinite(loss) and any(p.grad is not None and p.grad.abs().sum() > 0
                                        for p in model.parameters())


def test_summary_matches_evidence_index():
    root = Path(__file__).resolve().parents[1]
    records = json.loads((root / "results/summary.json").read_text())
    with (root / "results/summary.csv").open(newline="") as handle:
        table = list(csv.DictReader(handle))
    assert len(records) == len(table) and len(records) > 100
    assert all((root / row["evidence_path"]).is_file() for row in records)
    cache = {}

    def pointer(document, path):
        value = document
        for part in path.lstrip("/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            value = value[int(part)] if isinstance(value, list) else value[part]
        return value

    for row, csv_row in zip(records, table, strict=True):
        assert row["experiment_id"] == csv_row["experiment_id"]
        assert float(row["value"]) == float(csv_row["value"])
        source = root / row["evidence_path"]
        if source not in cache:
            cache[source] = json.loads(source.read_text())
        key = row["evidence_key"]
        if key.startswith("mean(") or key.startswith("sample_sd("):
            paths = key[key.index("(") + 1:-1].split(",")
            values = [pointer(cache[source], path) for path in paths]
            expected = statistics.mean(values) if key.startswith("mean(") else statistics.stdev(values)
        else:
            expected = pointer(cache[source], key)
        assert math.isclose(row["value"], expected, rel_tol=1e-10, abs_tol=1e-12)
    main = [row for row in records if row["experiment_id"] == "hard_iit"
            and row["split"] == "ood_test" and row["depth"] == "all"
            and row["metric"] == "answer_accuracy"]
    by_method = {row["method"]: row["value"] for row in main}
    assert 0.994 in by_method.values() and 0.504 in by_method.values()
