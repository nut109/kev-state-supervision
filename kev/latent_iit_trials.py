"""One-seed, rule-only hard-state trial after the gold and parser gates.

All learned-input arms start from the same *fresh* merged Kev initialization.
Only visible text enters the encoder; generator fields are loss/evaluation targets.
Locked tests require an explicit post-validation switch and are written once.
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
from torch import nn

from .latent_data import causal_intervention
from .latent_iit import HardRuleMachine
from .latent_iit_encoder import KEV_CHECKPOINT, OnePassKevRuleEncoder, rule_texts
from .latent_iit_run import pack_rules, rule_rows
from .release_utils import write_json


ARMS = ("deep", "r3", "r3_state", "r3_iit")
SEED = 11


def visible_fields(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Program length and option truth come only from source text/options."""
    depths = [len(rule_texts(row)[1]) for row in rows]
    options = []
    for row in rows:
        if set(row["options"]) != {"No", "Yes"}:
            raise ValueError("rule options must contain No and Yes")
        options.append([int(option == "Yes") for option in row["options"]])
    return (torch.tensor(depths, device=device),
            torch.tensor(options, device=device))


class TrialModel(nn.Module):
    def __init__(self, arm: str, device: str = "cpu", unfreeze_last: int = 6,
                 encoder: nn.Module | None = None):
        super().__init__()
        if arm not in (*ARMS, "gold"):
            raise ValueError(arm)
        self.arm = arm
        self.encoder = None if arm == "gold" else (
            encoder if encoder is not None else
            OnePassKevRuleEncoder(device=device, unfreeze_last=unfreeze_last))
        self.machine = HardRuleMachine("deep" if arm == "deep" else "recurrent",
                                       hidden=64, max_depth=10).to(device)

    def inputs(self, rows: list[dict]):
        if self.arm == "gold":
            target = pack_rules(rows)
            device = next(self.machine.parameters()).device
            initial = self.machine.gold_logits(target["initial"].to(device), 2)
            operators = self.machine.gold_logits(target["operators"].to(device), 4)
            gates = self.machine.gold_logits(target["gates"].to(device), 3)
        else:
            initial, operators, gates = self.encoder(rows)
            device = initial.device
        depths, options = visible_fields(rows, device)
        if (depths > operators.shape[1]).any() or (depths > self.machine.max_depth).any():
            raise ValueError("source rule count exceeds parsed program or machine depth")
        return initial, operators, gates, depths, options

    def forward(self, rows: list[dict], *, loops: int | None = None,
                return_step_logits: bool = False) -> dict:
        initial, operators, gates, depths, options = self.inputs(rows)
        loops = int(depths.max()) if loops is None else loops
        result = self.machine.rollout(initial, operators, gates, depths, options,
                                      loops=loops, return_step_logits=return_step_logits)
        return {"answer": result[0], "states": result[1],
                "step_logits": result[2] if return_step_logits else None,
                "parser": (initial, operators, gates), "depths": depths}


def trainable_state(model: TrialModel) -> dict:
    return {name: parameter.detach().cpu().clone()
            for name, parameter in model.encoder.named_parameters()
            if parameter.requires_grad}


def restore_trainable(model: TrialModel, saved: dict) -> None:
    parameters = dict(model.encoder.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            parameters[name].copy_(value.to(parameters[name].device))


def initialize(out: Path, device: str = "cuda", unfreeze_last: int = 6) -> None:
    for name in ("gold_rule_selection", "onepass_rule_selection", "onepass_rule_stress"):
        path = out / f"{name}.json"
        if not path.exists() or not json.loads(path.read_text())["passed"]:
            raise ValueError(f"required positive control has not passed: {path}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TrialModel("r3", device=device, unfreeze_last=unfreeze_last)
    torch.save({"seed": SEED, "kev_checkpoint": KEV_CHECKPOINT,
                "unfreeze_last": unfreeze_last,
                "encoder": trainable_state(model),
                "transition": {name: value.detach().cpu().clone()
                               for name, value in model.machine.transition.state_dict().items()}},
               out / "trial_common_init.pt")


def build_arm(out: Path, arm: str, device: str) -> TrialModel:
    if arm == "gold":
        model = TrialModel(arm, device=device)
        saved = torch.load(out / "gold_rule_best.pt", map_location="cpu", weights_only=True)
        model.machine.load_state_dict(saved["state_dict"])
        return model
    saved = torch.load(out / "trial_common_init.pt", map_location="cpu", weights_only=True)
    if saved["kev_checkpoint"] != KEV_CHECKPOINT:
        raise ValueError("Kev checkpoint differs from the saved common initialization")
    torch.manual_seed(SEED)
    model = TrialModel(arm, device=device, unfreeze_last=saved["unfreeze_last"])
    restore_trainable(model, saved["encoder"])
    if arm == "deep":
        for block in model.machine.transitions:
            block.load_state_dict(saved["transition"])
    else:
        model.machine.transition.load_state_dict(saved["transition"])
    return model


def eligible_interventions(rows: list[dict]) -> dict[int, list[tuple[str, str, int, int]]]:
    """Effectful, non-donor-copy tuples; generated fields never enter the model."""
    by_id = {row["id"]: row for row in rows}
    pool = defaultdict(list)
    for row in rows:
        donor = by_id[row["pair_example_id"]]
        depth = len(rule_texts(row)[1])
        for t in range(depth):
            result = causal_intervention(row, donor, t)
            if result["changed"] and not result["donor_copy"]:
                pool[depth].append((row["id"], donor["id"], t, result["label"]))
    if not pool:
        raise ValueError("no effectful non-donor-copy interventions")
    return dict(pool)


def state_loss(output: dict, rows: list[dict]) -> torch.Tensor:
    """Four equally weighted CE terms, using pre-hard transition logits."""
    initial, operators, gates = output["parser"]
    raw = output["step_logits"]
    device = initial.device
    target = pack_rules(rows)
    depths = output["depths"]
    if not torch.equal(depths.cpu(), target["depths"]):
        raise ValueError("source-derived depth disagrees with generator target")
    mask = torch.arange(raw.shape[1], device=device)[None, :] < depths[:, None]
    next_truth = torch.zeros(raw.shape[:2], dtype=torch.long, device=device)
    for i, trajectory in enumerate(target["trajectories"]):
        next_truth[i, :len(trajectory) - 1] = torch.tensor(trajectory[1:], device=device)
    losses = (
        F.cross_entropy(initial, target["initial"].to(device)),
        F.cross_entropy(operators[:, :raw.shape[1]][mask],
                        target["operators"][:, :raw.shape[1]].to(device)[mask]),
        F.cross_entropy(gates[:, :raw.shape[1]][mask],
                        target["gates"][:, :raw.shape[1]].to(device)[mask]),
        F.cross_entropy(raw[mask], next_truth[mask]),
    )
    return sum(losses) / 4


def train_batch(model: TrialModel, factual: list[dict],
                interventions: list[tuple[str, str, int, int]],
                by_id: dict[str, dict]) -> tuple[torch.Tensor, dict]:
    """Encode each distinct source row once; keep donor-state gradients in IIT."""
    merged = {row["id"]: row for row in factual}
    for recipient_id, donor_id, _, _ in interventions:
        merged[recipient_id] = by_id[recipient_id]
        merged[donor_id] = by_id[donor_id]
    rows = list(merged.values())
    at = {row["id"]: i for i, row in enumerate(rows)}
    initial, operators, gates, depths, options = model.inputs(rows)
    indices = torch.tensor([at[row["id"]] for row in factual], device=depths.device)
    loops = int(depths[indices].max())
    use_state = model.arm != "r3"
    output = model.machine.rollout(initial[indices], operators[indices], gates[indices],
                                   depths[indices], options[indices], loops=loops,
                                   return_step_logits=use_state)
    answer = F.cross_entropy(output[0], torch.tensor(
        [row["label"] for row in factual], device=depths.device))
    loss = answer
    terms = {"answer": float(answer.detach())}
    if use_state:
        supervised = state_loss({"parser": (initial[indices], operators[indices],
                                             gates[indices]), "step_logits": output[2],
                                 "depths": depths[indices]}, factual)
        loss = loss + supervised
        terms["state"] = float(supervised.detach())
    if interventions:
        cf_losses = []
        for recipient_id, donor_id, t, label in interventions:
            ri, di = at[recipient_id], at[donor_id]
            _, donor_states = model.machine.rollout(
                initial[di:di + 1], operators[di:di + 1], gates[di:di + 1],
                depths[di:di + 1], options[di:di + 1], loops=t)
            cf_logits, _ = model.machine.rollout(
                initial[ri:ri + 1], operators[ri:ri + 1], gates[ri:ri + 1],
                depths[ri:ri + 1], options[ri:ri + 1],
                loops=int(depths[ri]) - t, start_step=t,
                override_current=donor_states[:, t])
            cf_losses.append(F.cross_entropy(
                cf_logits, torch.tensor([label], device=depths.device)))
        intervention = torch.stack(cf_losses).mean()
        loss = loss + intervention
        terms["iit"] = float(intervention.detach())
    return loss, terms


@torch.no_grad()
def score_factual(model: TrialModel, rows: list[dict], batch_size: int,
                  *, curves: bool = False) -> dict:
    model.eval()
    totals = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    loop_totals = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    row_predictions = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        initial, operators, gates, depths, options = model.inputs(chunk)
        logits, states = model.machine.rollout(initial, operators, gates, depths,
                                               options, loops=int(depths.max()))
        predictions = logits.argmax(-1).tolist()
        paths = states.argmax(-1).tolist()
        nll = F.cross_entropy(logits, torch.tensor(
            [row["label"] for row in chunk], device=logits.device),
            reduction="none").tolist()
        parsed = (initial.argmax(-1).tolist(), operators.argmax(-1).tolist(),
                  gates.argmax(-1).tolist())
        target = pack_rules(chunk)
        for j, row in enumerate(chunk):
            depth = int(depths[j])
            row_predictions.append({"id": row["id"], "world_id": row["world_id"],
                                    "depth": depth, "label": row["label"],
                                    "prediction": predictions[j]})
            op = target["operators"][j, :depth].tolist()
            gate = target["gates"][j, :depth].tolist()
            op_pred = parsed[1][j][:depth]
            gate_pred = parsed[2][j][:depth]
            active = [i for i, value in enumerate(op) if value in (2, 3)]
            measures = {
                "answer": (int(predictions[j] == row["label"]), 1),
                "answer_nll": (nll[j], 1),
                "exact_trajectory": (int(paths[j][:depth + 1] ==
                                         target["trajectories"][j]), 1),
                "initial": (int(parsed[0][j] == target["initial"][j].item()), 1),
                "operator": (sum(a == b for a, b in zip(op_pred, op, strict=True)), depth),
                "gate": (sum(a == b for a, b in zip(gate_pred, gate, strict=True)), depth),
                "active_gate": (sum(gate_pred[i] == gate[i] for i in active), len(active)),
                "exact_program": (int(parsed[0][j] == target["initial"][j].item() and
                                      op_pred == op and gate_pred == gate), 1),
            }
            groups = ("all", f"depth_{depth}") + (("depth_6_8_10",) if depth in (6, 8, 10) else ())
            for group in groups:
                for name, (value, n) in measures.items():
                    totals[group][name][0] += value
                    totals[group][name][1] += n
        if curves:
            for loops in (3, 5, 8, 10):
                partial, _ = model.machine.rollout(initial, operators, gates, depths,
                                                   options, loops=loops)
                partial_pred = partial.argmax(-1).tolist()
                for j, depth in enumerate(depths.tolist()):
                    loop_totals[str(loops)][f"depth_{depth}"][0] += int(
                        partial_pred[j] == chunk[j]["label"])
                    loop_totals[str(loops)][f"depth_{depth}"][1] += 1
    return {"rows": row_predictions,
            "factual": {group: {name: {"value": value / n if n else None, "n": n}
                                for name, (value, n) in measures.items()}
                        for group, measures in totals.items()},
            "loops": {loop: {group: {"accuracy": good / n, "n": n}
                             for group, (good, n) in groups.items()}
                      for loop, groups in loop_totals.items()}}


@torch.no_grad()
def score_counterfactual(model: TrialModel, rows: list[dict],
                         worlds_per_batch: int = 4) -> dict:
    """Report true CF accuracy and prediction flips with separate denominators."""
    model.eval()
    worlds = defaultdict(list)
    for row in rows:
        worlds[row["world_id"]].append(row)
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
    pairs = list(worlds.values())
    for offset in range(0, len(pairs), worlds_per_batch):
        chunk = [row for pair in pairs[offset:offset + worlds_per_batch] for row in pair]
        at = {row["id"]: i for i, row in enumerate(chunk)}
        initial, operators, gates, depths, options = model.inputs(chunk)
        factual_logits, states = model.machine.rollout(
            initial, operators, gates, depths, options, loops=int(depths.max()))
        factual = factual_logits.argmax(-1).tolist()
        for i, row in enumerate(chunk):
            donor = chunk[at[row["pair_example_id"]]]
            di = at[donor["id"]]
            depth = int(depths[i])
            for t in range(depth):
                oracle = causal_intervention(row, donor, t)
                cf_logits, _ = model.machine.rollout(
                    initial[i:i + 1], operators[i:i + 1], gates[i:i + 1],
                    depths[i:i + 1], options[i:i + 1], loops=depth - t,
                    start_step=t, override_current=states[di:di + 1, t])
                prediction = int(cf_logits.argmax(-1).item())
                category = ("effectful_copy" if oracle["donor_copy"] else
                            "effectful_noncopy") if oracle["changed"] else "null"
                groups = ("all", f"depth_{depth}", f"step_{t}") + (
                    ("depth_6_8_10",) if depth in (6, 8, 10) else ())
                for group in groups:
                    record = totals[group][category]
                    record[0] += int(prediction == oracle["label"])
                    record[1] += int(prediction != factual[i])
                    record[2] += int(factual[i] == row["label"])
                    record[3] += 1
    return {group: {category: {"cf_consistency": good / n,
                               "flip_rate": flips / n,
                               "factual_accuracy": factual_correct / n,
                               "n": n}
                    for category, (good, flips, factual_correct, n) in categories.items()}
            for group, categories in totals.items()}


def fit(out: Path, arm: str, device: str, batch_size: int = 8, epochs: int = 4,
        *, smoke: bool = False) -> None:
    if arm not in ARMS:
        raise ValueError("only learned-input arms are fitted here")
    smoke_passed = None
    if not smoke:
        result = json.loads((out / f"trial_{arm}_smoke.json").read_text())
        smoke_passed = result["passed"]
        if not smoke_passed:
            print(f"{arm}: 64-row smoke failed; full fit is an optimization diagnostic", flush=True)
    rows = rule_rows(out, "train")
    if smoke:
        selected = set()
        for row in rows:
            selected.add(row["world_id"])
            if len(selected) == 32:
                break
        rows = [row for row in rows if row["world_id"] in selected]
        if len(rows) != 64:
            raise ValueError("smoke selection did not contain 32 complete pairs")
    val = rows if smoke else rule_rows(out, "val")
    by_id = {row["id"]: row for row in rows}
    pool = eligible_interventions(rows) if arm == "r3_iit" else {}
    pool_depths = sorted(pool)
    model = build_arm(out, arm, device)
    torch.manual_seed(SEED)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(SEED)
    groups = [
        {"params": [p for p in model.encoder.backbone.parameters() if p.requires_grad],
         "lr": 2e-5},
        {"params": [p for head in (model.encoder.initial_head,
                                    model.encoder.operator_head, model.encoder.gate_head)
                    for p in head.parameters()], "lr": 3e-4},
        {"params": list(model.machine.parameters()), "lr": 3e-3},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    order_rng = random.Random(SEED)
    cf_rng = random.Random(SEED + 1)
    best_key = (-1.0, float("-inf"))
    history = []
    cap = 80 if smoke else epochs
    for epoch in range(1, cap + 1):
        model.train()
        order = list(range(len(rows)))
        order_rng.shuffle(order)
        started = time.perf_counter()
        losses = []
        for offset in range(0, len(rows), batch_size):
            factual = [rows[i] for i in order[offset:offset + batch_size]]
            interventions = []
            if arm == "r3_iit":
                for _ in range(max(1, len(factual) // 2)):
                    depth = cf_rng.choice(pool_depths)
                    interventions.append(cf_rng.choice(pool[depth]))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = train_batch(model, factual, interventions, by_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters()
                                            if p.requires_grad), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        metrics = score_factual(model, val, batch_size)
        answer = metrics["factual"]["all"]["answer"]["value"]
        nll = metrics["factual"]["all"]["answer_nll"]["value"]
        key = (answer, -nll)
        record = {"epoch": epoch, "loss": sum(losses) / len(losses),
                  "seconds": time.perf_counter() - started,
                  "val_answer": answer, "val_nll": nll,
                  "val_exact_trajectory": metrics["factual"]["all"]["exact_trajectory"]["value"]}
        history.append(record)
        print({"arm": arm, **record}, flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_metrics = metrics
            if not smoke:
                torch.save({"arm": arm, "epoch": epoch,
                            "encoder": trainable_state(model),
                            "machine": {name: value.detach().cpu().clone()
                                        for name, value in model.machine.state_dict().items()}},
                           out / f"trial_{arm}_best.pt")
        if smoke and answer >= 0.95:
            break
    result = {"arm": arm, "seed": SEED, "kev_checkpoint": KEV_CHECKPOINT,
              "n_train": len(rows),
              "smoke_passed": smoke_passed,
              "batch": batch_size, "epochs_cap": cap,
              "learning_rates": {"backbone": 2e-5, "heads": 3e-4, "core": 3e-3},
              "trainable_parameters": sum(p.numel() for p in model.parameters()
                                          if p.requires_grad),
              "best_epoch": best_epoch, "best_metrics": best_metrics,
              "eligible_iit": {
                  str(depth): {"tuples": len(entries),
                               "recipient_rows": len({entry[0] for entry in entries}),
                               "worlds": len({by_id[entry[0]]["world_id"]
                                              for entry in entries})}
                  for depth, entries in pool.items()},
              "history": history}
    result["passed" if smoke else "fit_completed"] = (
        best_key[0] >= 0.95 if smoke else True)
    write_json(out / f"trial_{arm}_{'smoke' if smoke else 'selection'}.json", result)


def evaluate(out: Path, arm: str, split: str, device: str, batch_size: int,
             *, allow_locked_test: bool = False) -> None:
    locked = split in ("id_test", "ood_test")
    if locked and not allow_locked_test:
        raise ValueError("locked test requires --allow-locked-test after validation gate")
    if locked:
        for candidate in ARMS:
            selection = out / f"trial_{candidate}_selection.json"
            checkpoint = out / f"trial_{candidate}_best.pt"
            if (not selection.exists() or not checkpoint.exists() or
                    not json.loads(selection.read_text()).get("fit_completed")):
                raise ValueError(f"all four arms must be frozen before locked scoring: {candidate}")
    path = out / f"trial_{arm}_{split}.json"
    if locked and path.exists():
        raise ValueError(f"locked test was already scored: {path}")
    model = build_arm(out, arm, device)
    if arm != "gold":
        saved = torch.load(out / f"trial_{arm}_best.pt", map_location="cpu",
                           weights_only=True)
        restore_trainable(model, saved["encoder"])
        model.machine.load_state_dict(saved["machine"])
    rows = rule_rows(out, split)
    result = {"arm": arm, "split": split, "n": len(rows),
              "kev_checkpoint": None if arm == "gold" else KEV_CHECKPOINT,
              "trainable_parameters": sum(p.numel() for p in model.parameters()
                                          if p.requires_grad),
              **score_factual(model, rows, batch_size, curves=split != "val"),
              "counterfactual": score_counterfactual(model, rows)}
    write_json(path, result)
    print({"arm": arm, "split": split,
           "answer": result["factual"]["all"]["answer"],
           "effectful_noncopy": result["counterfactual"]["all"].get(
               "effectful_noncopy")}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("init", "smoke", "train", "evaluate"))
    parser.add_argument("--arm", choices=(*ARMS, "gold"))
    parser.add_argument("--split", choices=("val", "stress_val", "id_test", "ood_test"),
                        default="val")
    parser.add_argument("--out", type=Path, default=Path("runs/hard_iit"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--unfreeze-last", type=int, default=6)
    parser.add_argument("--allow-locked-test", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "init":
        initialize(args.out, args.device, args.unfreeze_last)
    elif args.stage in ("smoke", "train"):
        if args.arm is None:
            parser.error("--arm is required")
        fit(args.out, args.arm, args.device, args.batch, args.epochs,
            smoke=args.stage == "smoke")
    else:
        if args.arm is None:
            parser.error("--arm is required")
        evaluate(args.out, args.arm, args.split, args.device, args.batch,
                 allow_locked_test=args.allow_locked_test)


if __name__ == "__main__":
    main()
