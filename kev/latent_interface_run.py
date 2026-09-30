"""Paired state-interface ablation on the existing causal rule task.

The Kev parser is fitted once in the earlier experiment and frozen here. A,
B, and C receive the same cached parser logits, labels, initial weights, and
factual minibatches. Locked tests are opened only after all nine checkpoints
and the source hashes have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .latent_data import OPERATORS, causal_intervention, generate_causal_splits, solve_rules
from .latent_iit import HardRuleMachine
from .latent_iit_encoder import KEV_CHECKPOINT, OnePassKevRuleEncoder
from .latent_iit_run import pack_rules, rule_rows
from .latent_iit_trials import visible_fields
from .release_utils import paired_interval, write_json


ARMS = ("continuous", "soft", "hard")
SEEDS = (11, 23, 37)
DATA_SEED = 20261001
SIZES = {"train": 10_000, "val": 2_000, "stress_val": 200,
         "id_test": 2_000, "ood_test": 2_000}
TESTS = ("id_test", "ood_test")
MAX_DEPTH = 10


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sources() -> dict[str, Path]:
    directory = Path(__file__).resolve().parent
    return {name: directory / name for name in (
        "latent_interface_run.py", "latent_iit.py", "latent_iit_encoder.py",
        "latent_iit_run.py", "latent_iit_trials.py", "release_utils.py",
        "latent_data.py")}


def prepare(out: Path, parser_checkpoint: Path) -> None:
    if out.exists():
        raise ValueError(f"output directory already exists: {out}")
    if not parser_checkpoint.exists():
        raise ValueError(f"frozen parser checkpoint is missing: {parser_checkpoint}")
    out.mkdir(parents=True)
    splits = generate_causal_splits(DATA_SEED, SIZES)
    worlds = [{row["world_id"] for row in rows} for rows in splits.values()]
    if sum(map(len, worlds)) != len(set.union(*worlds)):
        raise ValueError("worlds overlap across splits")
    for split, rows in splits.items():
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                # Distinguish this fresh panel from the previously opened test.
                for key in ("id", "world_id", "pair_id", "pair_example_id"):
                    row[key] = f"interface-{DATA_SEED}-{row[key]}"
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "config.json", {
        "data_seed": DATA_SEED, "sizes_all_families": SIZES,
        "rule_rows": {name: len(rule_rows(out, name)) for name in splits},
        "kev_checkpoint": KEV_CHECKPOINT,
        "parser_checkpoint": str(parser_checkpoint.resolve()),
        "parser_sha256": digest(parser_checkpoint),
        "arms": ARMS, "seeds": SEEDS,
        "train_depths": [1, 2, 3], "ood_depths": [4, 5, 6, 8, 10],
        "batch": 128, "epochs": 30, "core_lr": 0.003,
        "single_repair": "Historical 12-epoch development gate failed for soft feedback; all nine arms were restarted with a 30-epoch cap before locked scoring.",
        "state_loss_weight": 0.25,
        "selection": "val answer accuracy, then val pre-feedback step CE, then answer NLL; stress is diagnostic only",
        "short_gate": "each seed/arm val answer >= 0.95 and per-seed arm spread <= 0.03, separately at depths 1/2/3",
        "long_decision": {
            "continue_interface": "B or C minus A >= 0.05 on depths 6/8/10 with paired world CI excluding zero, after the short gate; step errors and correct-direction counterfactuals must agree",
            "drop_hardness": "C minus B paired world CI wholly within [-0.02, 0.02] on depths 6/8/10",
            "stop_interface": "B minus A and C minus A paired world CIs wholly within [-0.02, 0.02] on depths 6/8/10",
        },
        "locked_test_policy": "freeze all nine selected checkpoints and code before reading either test",
    })


def _parser(out: Path, device: str) -> OnePassKevRuleEncoder:
    config = json.loads((out / "config.json").read_text())
    path = Path(config["parser_checkpoint"])
    if digest(path) != config["parser_sha256"]:
        raise ValueError("frozen parser checkpoint changed")
    selected = path.parent / "onepass_rule_selection.json"
    if not json.loads(selected.read_text())["passed"]:
        raise ValueError("frozen parser did not pass its validation gate")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = OnePassKevRuleEncoder(device=device, unfreeze_last=saved["unfreeze_last"])
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved["trainable"].items():
            parameters[name].copy_(value.to(parameters[name].device))
    return model.eval().requires_grad_(False)


@torch.inference_mode()
def cache_parser(out: Path, device: str, batch_size: int, *, locked: bool) -> None:
    if locked:
        verify_freeze(out)
    elif (out / "frozen_manifest.json").exists():
        raise ValueError("pre-test parser cache cannot change after freeze")
    destination = out / ("parser_cache_locked.pt" if locked else "parser_cache.pt")
    if destination.exists():
        raise ValueError(f"parser cache already exists: {destination}")
    model = _parser(out, device)
    cached = {}
    for split in (TESTS if locked else ("train", "val", "stress_val")):
        rows = rule_rows(out, split)
        parts = {name: [] for name in ("initial", "operators", "gates")}
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            visible = [{name: row[name] for name in ("question", "options", "state")}
                       for row in chunk]
            initial, operators, gates = model(visible)
            for name, tensor in (("initial", initial), ("operators", operators),
                                 ("gates", gates)):
                if tensor.ndim == 3:
                    tensor = F.pad(tensor, (0, 0, 0, MAX_DEPTH - tensor.shape[1]))
                parts[name].append(tensor.float().cpu())
        cached[split] = {"ids": [row["id"] for row in rows],
                         **{name: torch.cat(pieces) for name, pieces in parts.items()}}
        print({"cached": split, "rows": len(rows)}, flush=True)
    torch.save({"data_seed": DATA_SEED,
                "parser_sha256": json.loads((out / "config.json").read_text())["parser_sha256"],
                "splits": cached}, destination)
    if locked:
        write_json(out / "locked_cache_manifest.json", {
            "parser_sha256": json.loads((out / "config.json").read_text())["parser_sha256"],
            "cache_sha256": digest(destination)})


def load_split(out: Path, split: str) -> tuple[list[dict], dict]:
    cache_path = out / ("parser_cache_locked.pt" if split in TESTS else "parser_cache.pt")
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    config = json.loads((out / "config.json").read_text())
    rows = rule_rows(out, split)
    parsed = cache["splits"][split]
    if (cache["data_seed"] != DATA_SEED or
            cache["parser_sha256"] != config["parser_sha256"] or
            parsed["ids"] != [row["id"] for row in rows]):
        raise ValueError("parser cache does not match current rows")
    return rows, parsed


def machine(arm: str, seed: int) -> HardRuleMachine:
    torch.manual_seed(seed)
    return HardRuleMachine("recurrent", hidden=64, max_depth=MAX_DEPTH,
                           feedback=arm)


def _input(rows: list[dict], parsed: dict, indices) -> tuple:
    chosen = [rows[int(i)] for i in indices]
    initial = parsed["initial"][indices]
    operators = parsed["operators"][indices]
    gates = parsed["gates"][indices]
    depths, options = visible_fields(chosen, torch.device("cpu"))
    return initial, operators, gates, depths, options


def _targets(rows: list[dict], loops: int) -> torch.Tensor:
    target = torch.zeros((len(rows), loops), dtype=torch.long)
    for i, row in enumerate(rows):
        profile = next(item for item in row["rule_inputs"]
                       if item["subject"] == row["subject"])
        trajectory = solve_rules(profile["initial"], row["path"], profile["gates"])
        target[i, :len(trajectory)] = torch.tensor(trajectory, dtype=torch.long)
    return target


@torch.no_grad()
def short_score(model: HardRuleMachine, rows: list[dict], parsed: dict,
                batch: int) -> dict:
    model.eval()
    totals = defaultdict(lambda: [0, 0.0, 0, 0.0, 0])
    for start in range(0, len(rows), batch):
        indices = list(range(start, min(start + batch, len(rows))))
        selected = [rows[i] for i in indices]
        x = _input(rows, parsed, indices)
        logits, _, raw = model.rollout(*x, loops=int(x[3].max()),
                                       return_step_logits=True)
        labels = torch.tensor([row["label"] for row in selected])
        losses = F.cross_entropy(logits, labels, reduction="none")
        predictions = logits.argmax(-1)
        targets = _targets(selected, raw.shape[1])
        for j, row in enumerate(selected):
            depth = row["depth"]
            step_loss = float(F.cross_entropy(raw[j, :depth], targets[j, :depth],
                                              reduction="sum"))
            for key in ("all", f"depth_{row['depth']}"):
                record = totals[key]
                record[0] += int(predictions[j] == labels[j])
                record[1] += float(losses[j])
                record[2] += 1
                record[3] += step_loss
                record[4] += depth
    summary = {key: {"accuracy": good / n, "step_ce": step_loss / steps,
                     "nll": nll / n, "n": n}
               for key, (good, nll, n, step_loss, steps) in totals.items()}
    return {**summary["all"], "by_depth": {key: value for key, value in summary.items()
                                           if key != "all"}}


def train(out: Path) -> None:
    if (out / "frozen_manifest.json").exists():
        raise ValueError("checkpoints are frozen")
    train_rows, train_cache = load_split(out, "train")
    val_rows, val_cache = load_split(out, "val")
    stress_rows, stress_cache = load_split(out, "stress_val")
    config = json.loads((out / "config.json").read_text())
    for seed in SEEDS:
        for arm in ARMS:
            result_path = out / f"selection_{arm}_{seed}.json"
            if result_path.exists():
                continue
            model = machine(arm, seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["core_lr"],
                                          weight_decay=0.01)
            order_rng = random.Random(seed)
            best = (-1.0, float("-inf"))
            history = []
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                order = list(range(len(train_rows)))
                order_rng.shuffle(order)
                epoch_loss = []
                for start in range(0, len(order), config["batch"]):
                    indices = order[start:start + config["batch"]]
                    selected = [train_rows[i] for i in indices]
                    x = _input(train_rows, train_cache, indices)
                    answer, _, raw = model.rollout(*x, loops=int(x[3].max()),
                                                   return_step_logits=True)
                    labels = torch.tensor([row["label"] for row in selected])
                    mask = torch.arange(raw.shape[1])[None, :] < x[3][:, None]
                    state = F.cross_entropy(raw[mask], _targets(selected, raw.shape[1])[mask])
                    loss = F.cross_entropy(answer, labels) + config["state_loss_weight"] * state
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss.append(float(loss.detach()))
                val_score = short_score(model, val_rows, val_cache, config["batch"])
                record = {"epoch": epoch, "train_loss": sum(epoch_loss) / len(epoch_loss),
                          "val": val_score}
                history.append(record)
                key = (val_score["accuracy"], -val_score["step_ce"],
                       -val_score["nll"])
                if key > best:
                    best = key
                    best_epoch = epoch
                    torch.save({"arm": arm, "seed": seed, "epoch": epoch,
                                "state_dict": model.state_dict()},
                               out / f"best_{arm}_{seed}.pt")
                print({"arm": arm, "seed": seed, **record}, flush=True)
            saved = torch.load(out / f"best_{arm}_{seed}.pt", weights_only=True)
            model.load_state_dict(saved["state_dict"])
            stress = short_score(model, stress_rows, stress_cache, config["batch"])
            val_diagnostic = score(model, val_rows, val_cache, config["batch"],
                                   counterfactual=False)
            stress_diagnostic = score(model, stress_rows, stress_cache,
                                      config["batch"], counterfactual=False)
            write_json(result_path, {"arm": arm, "seed": seed, "best_epoch": best_epoch,
                                     "val": history[best_epoch - 1]["val"],
                                     "stress": stress, "history": history,
                                     "val_diagnostic": {
                                         key: val_diagnostic[key] for key in
                                         ("factual", "state_accuracy_by_step",
                                          "state_accuracy_by_depth_step",
                                          "first_error_distribution")},
                                     "stress_diagnostic": {
                                         key: stress_diagnostic[key] for key in
                                         ("factual", "state_accuracy_by_step",
                                          "state_accuracy_by_depth_step",
                                          "first_error_distribution")},
                                     "trainable_parameters": sum(p.numel() for p in model.parameters())})


def freeze(out: Path) -> None:
    path = out / "frozen_manifest.json"
    if path.exists():
        raise ValueError("experiment already frozen")
    config = json.loads((out / "config.json").read_text())
    for seed in SEEDS:
        values = defaultdict(list)
        for arm in ARMS:
            selection = out / f"selection_{arm}_{seed}.json"
            checkpoint = out / f"best_{arm}_{seed}.pt"
            if not selection.exists() or not checkpoint.exists():
                raise ValueError(f"missing selected model: {arm}/{seed}")
            val = json.loads(selection.read_text())["val"]
            for depth in (1, 2, 3):
                values[depth].append(val["by_depth"][f"depth_{depth}"]["accuracy"])
        for depth, accuracies in values.items():
            if min(accuracies) < 0.95 or max(accuracies) - min(accuracies) > 0.03:
                raise ValueError("short-depth comparability gate failed for "
                                 f"seed {seed}, depth {depth}: {accuracies}")
    files = {str(path.resolve()): digest(path) for path in (
        [out / "config.json", out / "parser_cache.pt"] +
        [out / f"{split}.jsonl" for split in ("train", "val", "stress_val", *TESTS)] +
        [out / f"best_{arm}_{seed}.pt" for seed in SEEDS for arm in ARMS] +
        [out / f"selection_{arm}_{seed}.json" for seed in SEEDS for arm in ARMS] +
        list(sources().values()) + [Path(config["parser_checkpoint"])])}
    write_json(path, {"data_seed": DATA_SEED, "files": files,
                      "selection_complete": True,
                      "decision_gate": config["short_gate"]})


def verify_freeze(out: Path) -> None:
    manifest = json.loads((out / "frozen_manifest.json").read_text())
    if manifest["data_seed"] != DATA_SEED:
        raise ValueError("wrong data seed")
    for name, expected in manifest["files"].items():
        if digest(Path(name)) != expected:
            raise ValueError(f"frozen input changed: {name}")


def exact_parser_answer(row: dict, initial: int, operators: list[int],
                        gates: list[int]) -> int:
    gate_values = [None if value == 0 else value == 2 for value in gates]
    truth = solve_rules(bool(initial), [OPERATORS[i] for i in operators], gate_values)[-1]
    return row["options"].index("Yes" if truth else "No")


@torch.no_grad()
def score(model: HardRuleMachine, rows: list[dict], parsed: dict,
          batch: int, *, counterfactual: bool = True) -> dict:
    model.eval()
    if counterfactual:
        if batch % 2:
            raise ValueError("counterfactual batch must contain complete paired worlds")
        worlds = defaultdict(list)
        for i, row in enumerate(rows):
            worlds[row["world_id"]].append(i)
        if any(len(indices) != 2 for indices in worlds.values()):
            raise ValueError("each counterfactual world must have two rows")
        order = [i for indices in worlds.values() for i in indices]
    else:
        order = list(range(len(rows)))
    per_row = []
    by_depth = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    per_step = defaultdict(lambda: [0, 0])
    by_depth_step = defaultdict(lambda: [0, 0])
    first_error = defaultdict(lambda: defaultdict(int))
    cf = defaultdict(lambda: [0, 0, 0, 0])
    for start in range(0, len(order), batch):
        indices = order[start:start + batch]
        chunk = [rows[i] for i in indices]
        x = _input(rows, parsed, indices)
        logits, states = model.rollout(*x, loops=int(x[3].max()))
        target = pack_rules(chunk)
        gold_x = (model.gold_logits(target["initial"], 2),
                  model.gold_logits(target["operators"], 4),
                  model.gold_logits(target["gates"], 3), x[3], x[4])
        gold_logits, _ = model.rollout(*gold_x, loops=int(x[3].max()))
        initial = x[0].argmax(-1).tolist()
        operators = x[1].argmax(-1).tolist()
        gates = x[2].argmax(-1).tolist()
        predictions = logits.argmax(-1).tolist()
        gold_predictions = gold_logits.argmax(-1).tolist()
        paths = states.argmax(-1).tolist()
        exact_flags = []
        for j, row in enumerate(chunk):
            depth = row["depth"]
            gold_path = target["trajectories"][j]
            exact = (initial[j] == gold_path[0] and
                     operators[j][:depth] == target["operators"][j, :depth].tolist() and
                     gates[j][:depth] == target["gates"][j, :depth].tolist())
            exact_flags.append(exact)
            oracle = exact_parser_answer(row, initial[j], operators[j][:depth],
                                         gates[j][:depth])
            errors = [t for t in range(depth + 1) if paths[j][t] != gold_path[t]]
            first = errors[0] if errors else None
            first_key = str(first) if first is not None else "none"
            first_error[f"depth_{depth}"][first_key] += 1
            first_error["all"][first_key] += 1
            recovered = (first is not None and
                         any(paths[j][t] == gold_path[t]
                             for t in range(first + 1, depth + 1)))
            item = {"id": row["id"], "world_id": row["world_id"], "depth": depth,
                    "label": row["label"], "pred": predictions[j],
                    "gold_input_executor_pred": gold_predictions[j],
                    "exact_executor_pred": oracle, "parser_exact": exact,
                    "path": paths[j][:depth + 1], "gold_path": gold_path,
                    "first_error": first, "recovered": recovered}
            per_row.append(item)
            for key in ("all", f"depth_{depth}"):
                measures = by_depth[key]
                for name, value in (("answer", predictions[j] == row["label"]),
                                    ("answer_parser_exact", exact and predictions[j] == row["label"]),
                                    ("parser_exact", exact),
                                    ("gold_input_executor_answer", gold_predictions[j] == row["label"]),
                                    ("exact_executor_answer", oracle == row["label"]),
                                    ("exact_trajectory", not errors),
                                    ("any_error", bool(errors)),
                                    ("recovered", recovered)):
                    measures[name][0] += int(value)
                    measures[name][1] += (int(exact) if name == "answer_parser_exact"
                                          else int(bool(errors)) if name == "recovered" else 1)
                gold_ops = target["operators"][j, :depth].tolist()
                gold_gates = target["gates"][j, :depth].tolist()
                active = [t for t, op in enumerate(gold_ops) if op in (2, 3)]
                for name, good, n in (
                    ("initial", int(initial[j] == gold_path[0]), 1),
                    ("operator", sum(a == b for a, b in zip(
                        operators[j][:depth], gold_ops, strict=True)), depth),
                    ("gate", sum(a == b for a, b in zip(
                        gates[j][:depth], gold_gates, strict=True)), depth),
                    ("active_gate", sum(gates[j][t] == gold_gates[t]
                                        for t in active), len(active)),
                ):
                    measures[name][0] += good
                    measures[name][1] += n
            for t in range(depth + 1):
                per_step[t][0] += int(paths[j][t] == gold_path[t])
                per_step[t][1] += 1
                by_depth_step[(depth, t)][0] += int(paths[j][t] == gold_path[t])
                by_depth_step[(depth, t)][1] += 1
        if not counterfactual:
            continue
        at = {row["id"]: j for j, row in enumerate(chunk)}
        for j, row in enumerate(chunk):
            donor_at = at.get(row["pair_example_id"])
            if donor_at is None:
                raise ValueError("a counterfactual batch must contain complete paired worlds")
            donor = chunk[donor_at]
            depth = row["depth"]
            for t in range(depth):
                oracle = causal_intervention(row, donor, t)
                cf_logits, _ = model.rollout(x[0][j:j + 1], x[1][j:j + 1],
                                             x[2][j:j + 1], x[3][j:j + 1],
                                             x[4][j:j + 1], loops=depth - t,
                                             start_step=t,
                                             override_current=states[donor_at:donor_at + 1, t])
                prediction = int(cf_logits.argmax(-1).item())
                if oracle["changed"]:
                    category = "effectful_copy" if oracle["donor_copy"] else "effectful_noncopy"
                elif oracle["recipient_value"] == oracle["donor_value"]:
                    category = "null_same_state"
                else:
                    category = "null_erased"
                categories = [category]
                if category == "null_same_state" and t:
                    own_profile = next(p for p in row["rule_inputs"]
                                       if p["subject"] == row["subject"])
                    donor_profile = next(p for p in donor["rule_inputs"]
                                         if p["subject"] == donor["subject"])
                    if (own_profile["initial"] != donor_profile["initial"] or
                            row["path"][:t] != donor["path"][:t] or
                            own_profile["gates"][:t] != donor_profile["gates"][:t]):
                        categories.append("null_same_state_diff_prefix")
                if (exact_flags[j] and
                        int(states[donor_at, t].argmax()) == target["trajectories"][donor_at][t]):
                    categories.extend(f"{name}_conditioned" for name in categories.copy())
                for key in ("all", f"depth_{depth}"):
                    for subset in categories:
                        record = cf[(key, subset)]
                        record[0] += int(prediction == oracle["label"])
                        record[1] += int(prediction != predictions[j])
                        record[2] += int(prediction == oracle["label"] and
                                         prediction != predictions[j])
                        record[3] += 1
    metrics = {key: {name: {"value": hits / n if n else None, "n": n}
                     for name, (hits, n) in measures.items()}
               for key, measures in by_depth.items()}
    return {"rows": per_row, "factual": metrics,
            "first_error_distribution": {key: dict(counts)
                                         for key, counts in first_error.items()},
            "state_accuracy_by_step": {str(t): {"accuracy": good / n, "n": n}
                                       for t, (good, n) in sorted(per_step.items())},
            "state_accuracy_by_depth_step": {
                f"depth_{depth}/step_{t}": {"accuracy": good / n, "n": n}
                for (depth, t), (good, n) in sorted(by_depth_step.items())},
            "counterfactual_condition": "conditioned means recipient parser exact and donor predicted state matches gold at swap step",
            "counterfactual": {f"{key}/{category}": {
                "consistency": good / n, "flip_rate": flips / n,
                "correct_direction_flip": correct_flip / n, "n": n}
                for (key, category), (good, flips, correct_flip, n) in cf.items()}}


def evaluate(out: Path, batch: int) -> None:
    verify_freeze(out)
    result_path = out / "locked_results.json"
    if result_path.exists() or (out / "locked_started.json").exists():
        raise ValueError("locked tests were already opened")
    if not (out / "parser_cache_locked.pt").exists():
        raise ValueError("locked parser cache is missing")
    cache_manifest = json.loads((out / "locked_cache_manifest.json").read_text())
    if (digest(out / "parser_cache_locked.pt") != cache_manifest["cache_sha256"] or
            cache_manifest["parser_sha256"] !=
            json.loads((out / "config.json").read_text())["parser_sha256"]):
        raise ValueError("locked parser cache changed")
    write_json(out / "locked_started.json", {
        "data_seed": DATA_SEED, "cache_sha256": cache_manifest["cache_sha256"]})
    results = {}
    for split in TESTS:
        rows, parsed = load_split(out, split)
        results[split] = {}
        for seed in SEEDS:
            for arm in ARMS:
                model = machine(arm, seed)
                saved = torch.load(out / f"best_{arm}_{seed}.pt", weights_only=True)
                model.load_state_dict(saved["state_dict"])
                results[split][f"{arm}_{seed}"] = score(model, rows, parsed, batch)
                print({"split": split, "arm": arm, "seed": seed,
                       "answer": results[split][f"{arm}_{seed}"]["factual"]["all"]["answer"]},
                      flush=True)
        if split == "ood_test":
            results["paired_intervals"] = {}
            for a, b in (("soft", "continuous"), ("hard", "soft"),
                         ("hard", "continuous")):
                for depths in (None, (6, 8, 10)):
                    def paired_rows(arm):
                        return [[row for row in results[split][f"{arm}_{seed}"]["rows"]
                                 if depths is None or row["depth"] in depths]
                                for seed in SEEDS]
                    key = f"{a}_minus_{b}" + ("_late" if depths else "_all")
                    results["paired_intervals"][key] = paired_interval(
                        paired_rows(a), paired_rows(b), seed=DATA_SEED,
                        replicates=10_000)
    write_json(result_path, results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "cache", "train", "freeze",
                                          "cache_locked", "evaluate"))
    parser.add_argument("--out", type=Path, default=Path("runs/state_interface"))
    parser.add_argument("--parser-checkpoint", type=Path,
                        default=Path("runs/hard_iit/onepass_rule_best.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.out, args.parser_checkpoint)
    elif args.stage == "cache":
        cache_parser(args.out, args.device, args.batch, locked=False)
    elif args.stage == "train":
        train(args.out)
    elif args.stage == "freeze":
        freeze(args.out)
    elif args.stage == "cache_locked":
        cache_parser(args.out, args.device, args.batch, locked=True)
    else:
        evaluate(args.out, args.batch)


if __name__ == "__main__":
    main()
