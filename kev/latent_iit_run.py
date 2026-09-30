"""Staged hard-state/IIT diagnosis on fresh depth-extrapolation worlds.

Gold transitions and the Kev parser are checked on separate shallow validation
and long-depth stress sets. Locked test partitions are never read by these gates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .latent_data import (OPERATORS, causal_intervention, generate_causal_splits,
                          solve_rules)
from .latent_iit import HardRuleMachine
from .release_utils import read_rows, write_json


def prepare(out: Path, seed: int = 20260929) -> None:
    out.mkdir(parents=True, exist_ok=True)
    splits = generate_causal_splits(seed)
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(out / "config.json", {
        "data_seed": seed, "sizes": {name: len(rows) for name, rows in splits.items()},
        "train_depths": [1, 2, 3], "stress_and_ood_depths": [4, 5, 6, 8, 10],
        "gold_core": {"mode": "recurrent", "hidden": 64, "lr": 0.003,
                      "batch": 256, "epochs": 60, "selection": "validation answer accuracy"},
        "locked_test_policy": "do not score id_test or ood_test before validation gate",
    })


def rule_rows(out: Path, split: str) -> list[dict]:
    return [row for row in read_rows(out / f"{split}.jsonl") if row["family"] == "rule"]


def pack_rules(rows: list[dict], max_depth: int = 10) -> dict[str, torch.Tensor]:
    """Gold input comes from primitives; traces are recomputed only for metrics."""
    initial, operators, gates, depths, options, labels, trajectories = [], [], [], [], [], [], []
    for row in rows:
        profile = next(item for item in row["rule_inputs"] if item["subject"] == row["subject"])
        depth = row["depth"]
        initial.append(int(profile["initial"]))
        operators.append([OPERATORS.index(value) for value in row["path"]] +
                         [0] * (max_depth - depth))
        gates.append([0 if value is None else 2 if value else 1
                      for value in profile["gates"]] + [0] * (max_depth - depth))
        depths.append(depth)
        options.append([int(value == "Yes") for value in row["options"]])
        labels.append(row["label"])
        trajectories.append([int(profile["initial"])] +
                            [int(value) for value in solve_rules(
                                profile["initial"], row["path"], profile["gates"])])
    return {"initial": torch.tensor(initial), "operators": torch.tensor(operators),
            "gates": torch.tensor(gates), "depths": torch.tensor(depths),
            "options": torch.tensor(options), "labels": torch.tensor(labels),
            "trajectories": trajectories}


def gold_forward(model: HardRuleMachine, batch: dict, index, loops: int):
    initial = model.gold_logits(batch["initial"][index], 2)
    operators = model.gold_logits(batch["operators"][index], 4)
    gates = model.gold_logits(batch["gates"][index], 3)
    return model.rollout(initial, operators, gates, batch["depths"][index],
                         batch["options"][index], loops=loops)


@torch.no_grad()
def score(model: HardRuleMachine, rows: list[dict], loops: int) -> dict:
    model.eval()
    batch = pack_rules(rows)
    groups = defaultdict(lambda: [0, 0])
    trajectory = defaultdict(lambda: [0, 0])
    for start in range(0, len(rows), 256):
        indices = slice(start, min(start + 256, len(rows)))
        logits, states = gold_forward(model, batch, indices, loops)
        predictions = logits.argmax(-1).tolist()
        paths = states.argmax(-1).tolist()
        for j, pred in enumerate(predictions, start):
            row = rows[j]
            depth = row["depth"]
            for key in ("all", f"depth_{depth}"):
                groups[key][0] += int(pred == row["label"])
                groups[key][1] += 1
                steps = min(depth, loops) + 1
                trajectory[key][0] += int(paths[j - start][:steps] ==
                                           batch["trajectories"][j][:steps])
                trajectory[key][1] += 1
    return {"answer": {key: {"accuracy": yes / total, "n": total}
                       for key, (yes, total) in groups.items()},
            "exact_observed_trajectory": {key: {"accuracy": yes / total, "n": total}
                                          for key, (yes, total) in trajectory.items()}}


def fit_gold(out: Path, *, smoke: bool) -> None:
    random.seed(11)
    torch.manual_seed(11)
    rows = rule_rows(out, "train")
    val = rule_rows(out, "val")
    if smoke:
        rows = rows[:64]
    batch = pack_rules(rows)
    model = HardRuleMachine("recurrent", hidden=64, max_depth=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    best, history = -1.0, []
    cap = 120 if smoke else 60
    for epoch in range(1, cap + 1):
        model.train()
        order = torch.randperm(len(rows))
        for indices in order.split(64 if smoke else 256):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = gold_forward(model, batch, indices, 3)
            loss = F.cross_entropy(logits, batch["labels"][indices])
            loss.backward()
            optimizer.step()
        train_score = score(model, rows, 3)["answer"]["all"]["accuracy"]
        val_score = None if smoke else score(model, val, 3)["answer"]["all"]["accuracy"]
        history.append({"epoch": epoch, "train_accuracy": train_score,
                        "val_accuracy": val_score})
        selection = train_score if smoke else val_score
        if selection > best:
            best = selection
            if not smoke:
                torch.save({"state_dict": model.state_dict(), "best_epoch": epoch,
                            "val_accuracy": val_score}, out / "gold_rule_best.pt")
        if best >= 0.98:
            break
    result = {"seed": 11, "n_train": len(rows), "best_accuracy": best,
              "passed": best >= (0.98 if smoke else 0.95),
              "history": history}
    write_json(out / ("gold_rule_smoke.json" if smoke else "gold_rule_selection.json"), result)
    print({key: result[key] for key in ("n_train", "best_accuracy", "passed")}, flush=True)


@torch.no_grad()
def gold_stress(out: Path) -> None:
    selection = json.loads((out / "gold_rule_selection.json").read_text())
    if not selection["passed"]:
        raise ValueError("gold shallow validation gate failed")
    saved = torch.load(out / "gold_rule_best.pt", map_location="cpu", weights_only=True)
    model = HardRuleMachine("recurrent", hidden=64, max_depth=10)
    model.load_state_dict(saved["state_dict"])
    rows = rule_rows(out, "stress_val")
    by_id = {row["id"]: row for row in read_rows(out / "stress_val.jsonl")}
    curves = {str(loops): score(model, rows, loops)
              for loops in (3, 5, 8, 10)}
    totals = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        donor = by_id[row["pair_example_id"]]
        a, b = pack_rules([row]), pack_rules([donor])
        for t in range(row["depth"]):
            oracle = causal_intervention(row, donor, t)
            _, donor_states = gold_forward(model, b, slice(None), t)
            logits, _ = model.rollout(
                model.gold_logits(a["initial"], 2),
                model.gold_logits(a["operators"], 4),
                model.gold_logits(a["gates"], 3),
                a["depths"], a["options"], loops=row["depth"] - t,
                start_step=t, override_current=donor_states[:, t])
            key = ("effectful_copy" if oracle["donor_copy"] else "effectful_noncopy") \
                if oracle["changed"] else "null"
            totals[key][0] += int(logits.argmax(-1).item() == oracle["label"])
            totals[key][1] += 1
            totals[key][2] += int(oracle["donor_copy"])
    write_json(out / "gold_rule_stress.json", {
        "curves": curves,
        "interventions": {key: {"cf_consistency": good / n, "n": n,
                                 "donor_copy_cases": copies}
                          for key, (good, n, copies) in totals.items()},
    })


def parser_loss(logits: tuple[torch.Tensor, ...], rows: list[dict]) -> torch.Tensor:
    """Supervise typed factors at real steps, ignoring batch padding."""
    initial, operators, gates = logits
    target = pack_rules(rows)
    device = initial.device
    depth = target["depths"].to(device)
    mask = torch.arange(operators.shape[1], device=device)[None] < depth[:, None]
    truth = target["initial"].to(device)
    op = target["operators"][:, :operators.shape[1]].to(device)
    gate = target["gates"][:, :gates.shape[1]].to(device)
    return (F.cross_entropy(initial, truth) +
            F.cross_entropy(operators[mask], op[mask]) +
            F.cross_entropy(gates[mask], gate[mask]))


@torch.no_grad()
def score_parser(model, rows: list[dict], batch_size: int) -> dict:
    """Initial, step, nontrivial-gate, and exact-program accuracy by depth."""
    model.eval()
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        initial, operators, gates = model(chunk)
        predictions = (initial.argmax(-1).tolist(),
                       operators.argmax(-1).tolist(), gates.argmax(-1).tolist())
        target = pack_rules(chunk)
        for j, row in enumerate(chunk):
            depth = row["depth"]
            op = target["operators"][j, :depth].tolist()
            gate = target["gates"][j, :depth].tolist()
            op_pred = predictions[1][j][:depth]
            gate_pred = predictions[2][j][:depth]
            active = [i for i, value in enumerate(op) if value in (2, 3)]
            measures = {
                "initial": (int(predictions[0][j] == target["initial"][j].item()), 1),
                "operator": (sum(a == b for a, b in zip(op_pred, op, strict=True)), depth),
                "gate": (sum(a == b for a, b in zip(gate_pred, gate, strict=True)), depth),
                "active_gate": (sum(gate_pred[i] == gate[i] for i in active), len(active)),
                "exact_program": (int(predictions[0][j] == target["initial"][j].item() and
                                      op_pred == op and gate_pred == gate), 1),
            }
            for group in ("all", f"depth_{depth}"):
                for name, (correct, n) in measures.items():
                    totals[group][name][0] += correct
                    totals[group][name][1] += n
    return {group: {name: {"accuracy": correct / n if n else None, "n": n}
                    for name, (correct, n) in measures.items()}
            for group, measures in totals.items()}


def parser_gate(metrics: dict, threshold: float) -> bool:
    return all(metrics["all"][name]["accuracy"] is not None and
               metrics["all"][name]["accuracy"] >= threshold
               for name in ("initial", "operator", "gate", "active_gate"))


def fit_parser(out: Path, *, smoke: bool, batch_size: int, epochs: int | None,
               unfreeze_last: int, lr_backbone: float, lr_head: float,
               device: str, one_pass: bool = False) -> None:
    """Tune only Kev's final blocks and three shared categorical heads."""
    from .latent_iit_encoder import KevRuleEncoder, OnePassKevRuleEncoder

    prefix = "onepass_rule" if one_pass else "parser_rule"
    encoder = OnePassKevRuleEncoder if one_pass else KevRuleEncoder
    random.seed(29)
    torch.manual_seed(29)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(29)
    if not smoke:
        result = json.loads((out / f"{prefix}_smoke.json").read_text())
        if not result["passed"]:
            raise ValueError("parser smoke gate failed")
    rows = rule_rows(out, "train")
    if smoke:
        rows = rows[:64]
    else:
        val = rule_rows(out, "val")
    model = encoder(device=device, unfreeze_last=unfreeze_last)
    groups = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr": lr_backbone},
        {"params": [p for head in (model.initial_head, model.operator_head,
                                    model.gate_head) for p in head.parameters()],
         "lr": lr_head},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    history = []
    best_key = (-1.0, -1.0)
    cap = epochs if epochs is not None else (20 if smoke else 4)
    for epoch in range(1, cap + 1):
        model.train()
        order = torch.randperm(len(rows)).tolist()
        started = time.perf_counter()
        losses = []
        for offset in range(0, len(rows), batch_size):
            chunk = [rows[i] for i in order[offset:offset + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = parser_loss(model(chunk), chunk)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()
                                            if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        metrics = score_parser(model, rows if smoke else val, batch_size)
        factors = metrics["all"]
        key = (min(factors[name]["accuracy"] for name in
                   ("initial", "operator", "gate", "active_gate")),
               factors["exact_program"]["accuracy"])
        record = {"epoch": epoch, "loss": sum(losses) / len(losses),
                  "seconds": time.perf_counter() - started,
                  "selection_metrics": factors}
        history.append(record)
        print({"epoch": epoch, "loss": round(record["loss"], 4),
               "seconds": round(record["seconds"], 1),
               "initial": round(factors["initial"]["accuracy"], 3),
               "operator": round(factors["operator"]["accuracy"], 3),
               "active_gate": round(factors["active_gate"]["accuracy"], 3),
               "exact_program": round(factors["exact_program"]["accuracy"], 3)},
              flush=True)
        if key > best_key:
            best_key = key
            best_metrics = metrics
            best_epoch = epoch
            if not smoke:
                trainable = {name: parameter.detach().cpu().clone()
                             for name, parameter in model.named_parameters()
                             if parameter.requires_grad}
                torch.save({"trainable": trainable, "unfreeze_last": unfreeze_last,
                            "best_epoch": epoch, "one_pass": one_pass},
                           out / f"{prefix}_best.pt")
        if (smoke and factors["exact_program"]["accuracy"] >= 0.95) or \
                (not smoke and parser_gate(metrics, 0.90)):
            break
    result = {
        "seed": 29, "readout_mode": "one_pass" if one_pass else "source_local",
        "n_train": len(rows),
        "n_selection": len(rows) if smoke else len(val),
        "selection_split": "train" if smoke else "val",
        "batch": batch_size, "epochs_cap": cap, "unfreeze_last": unfreeze_last,
        "lr_backbone": lr_backbone, "lr_head": lr_head,
        "best_epoch": best_epoch, "best_metrics": best_metrics,
        "passed": (best_metrics["all"]["exact_program"]["accuracy"] >= 0.95
                   if smoke else parser_gate(best_metrics, 0.90)),
        "history": history,
    }
    write_json(out / f"{prefix}_{'smoke' if smoke else 'selection'}.json", result)
    print({"best_epoch": best_epoch, "passed": result["passed"],
           "best_metrics": best_metrics["all"]}, flush=True)


@torch.no_grad()
def parser_stress(out: Path, *, batch_size: int, device: str,
                  one_pass: bool = False) -> None:
    from .latent_iit_encoder import KevRuleEncoder, OnePassKevRuleEncoder

    prefix = "onepass_rule" if one_pass else "parser_rule"
    selection = json.loads((out / f"{prefix}_selection.json").read_text())
    if not selection["passed"]:
        raise ValueError("parser shallow validation gate failed")
    saved = torch.load(out / f"{prefix}_best.pt", map_location="cpu",
                       weights_only=True)
    encoder = OnePassKevRuleEncoder if one_pass else KevRuleEncoder
    model = encoder(device=device, unfreeze_last=saved["unfreeze_last"])
    parameters = dict(model.named_parameters())
    for name, value in saved["trainable"].items():
        parameters[name].copy_(value.to(parameters[name].device))
    metrics = score_parser(model, rule_rows(out, "stress_val"), batch_size)
    result = {"selection_best_epoch": saved["best_epoch"],
              "metrics": metrics, "passed": parser_gate(metrics, 0.80)}
    write_json(out / f"{prefix}_stress.json", result)
    print({"passed": result["passed"], "metrics": metrics["all"]}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "gold_smoke", "gold_train",
                                          "gold_stress", "parser_smoke",
                                          "parser_train", "parser_stress",
                                          "onepass_smoke", "onepass_train",
                                          "onepass_stress"))
    parser.add_argument("--out", type=Path, default=Path("runs/hard_iit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--unfreeze-last", type=int, default=6)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "prepare":
        prepare(args.out)
    elif args.stage == "gold_smoke":
        fit_gold(args.out, smoke=True)
    elif args.stage == "gold_train":
        fit_gold(args.out, smoke=False)
    elif args.stage == "gold_stress":
        gold_stress(args.out)
    else:
        one_pass = args.stage.startswith("onepass")
        batch = args.batch or (16 if one_pass else 8)
        if args.stage.endswith("_stress"):
            parser_stress(args.out, batch_size=batch, device=args.device,
                          one_pass=one_pass)
        else:
            fit_parser(args.out, smoke=args.stage.endswith("_smoke"),
                       batch_size=batch, epochs=args.epochs,
                       unfreeze_last=args.unfreeze_last,
                       lr_backbone=args.lr_backbone, lr_head=args.lr_head,
                       device=args.device, one_pass=one_pass)


if __name__ == "__main__":
    main()
