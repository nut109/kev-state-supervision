"""Gold-state support-hole study of interchange intervention training.

The only learned component is a shared finite-state transition. Factual paths
contain seen (state, rule) pairs; all intervention donors are factual paths.
IIT and direct counterfactual augmentation consume the same target-hole tuples.
Run ``python -m kev.latent_support_hole --smoke`` before a full run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


STATES, RULES = 16, 8
ARMS = ("State", "State+IIT", "State+CF-Aug", "State+IIT-seen")


@dataclass(frozen=True)
class Program:
    initial: int
    rules: tuple[int, ...]
    trace: tuple[int, ...]


@dataclass(frozen=True)
class Intervention:
    recipient: int
    donor: int
    seen_donor: int
    step: int
    hole_state: int
    seen_state: int


class HardTransition(nn.Module):
    """One recurrent transition with a hard categorical state interface."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(STATES + RULES, hidden), nn.GELU(),
                                 nn.Linear(hidden, STATES))

    def forward(self, state: torch.Tensor, rule: torch.Tensor) -> torch.Tensor:
        state = F.one_hot(state.long(), STATES).float() if state.ndim == 1 else state
        return self.net(torch.cat((state, F.one_hot(rule.long(), RULES).float()), -1))


def hard_state(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(-1)
    one_hot = F.one_hot(probs.argmax(-1), STATES).to(probs.dtype)
    return one_hot + probs - probs.detach()


def machine(seed: int) -> tuple[tuple[tuple[int, ...], ...], frozenset[tuple[int, int]]]:
    rng = random.Random(seed)
    permutations = []
    for _ in range(RULES):
        values = list(range(STATES))
        rng.shuffle(values)
        permutations.append(values)
    table = tuple(tuple(permutations[r][s] for r in range(RULES))
                  for s in range(STATES))
    assert all(sorted(table[s][r] for s in range(STATES)) == list(range(STATES))
               for r in range(RULES))
    pairs = [(s, r) for s in range(STATES) for r in range(RULES)]
    while True:
        holes = frozenset(rng.sample(pairs, 38))
        if (all(any((s, r) in holes for r in range(RULES)) and
                any((s, r) not in holes for r in range(RULES)) for s in range(STATES))
                and all(any((s, r) in holes for s in range(STATES)) and
                        any((s, r) not in holes for s in range(STATES))
                        for r in range(RULES))):
            return table, holes


def traced(initial: int, rules: tuple[int, ...], table) -> Program:
    states = [initial]
    for rule in rules:
        states.append(table[states[-1]][rule])
    return Program(initial, rules, tuple(states))


def factual_programs(table, holes, seed: int) -> tuple[list[Program], list[Program]]:
    rows = []

    def extend(initial: int, state: int, rules: tuple[int, ...], trace: tuple[int, ...]):
        if rules:
            rows.append(Program(initial, rules, trace))
        if len(rules) < 3:
            for rule in range(RULES):
                if (state, rule) not in holes:
                    next_state = table[state][rule]
                    extend(initial, next_state, rules + (rule,), trace + (next_state,))

    for state in range(STATES):
        extend(state, state, (), (state,))
    random.Random(seed).shuffle(rows)
    cut = int(0.8 * len(rows))
    train, val = rows[:cut], rows[cut:]
    keys = lambda part: {(p.initial, p.rules) for p in part}
    assert keys(train).isdisjoint(keys(val))
    assert all((state, rule) not in holes for part in (train, val)
               for p in part for state, rule in zip(p.trace, p.rules))
    seen = {(state, rule) for p in train
            for state, rule in zip(p.trace, p.rules)}
    assert seen == {(s, r) for s in range(STATES) for r in range(RULES)} - holes
    return train, val


def suffix_is_seen(program: Program, step: int, injected: int, table, holes) -> bool:
    state = table[injected][program.rules[step]]
    for rule in program.rules[step + 1:]:
        if (state, rule) in holes:
            return False
        state = table[state][rule]
    return True


def interventions(train: list[Program], table, holes, seed: int,
                  per_hole: int) -> list[Intervention]:
    rng = random.Random(seed)
    positions = defaultdict(list)
    donors = defaultdict(list)
    for index, program in enumerate(train):
        for step, rule in enumerate(program.rules):
            positions[rule].append((index, step))
        for step in range(len(program.rules) + 1):
            donors[(step, program.trace[step])].append(index)
    result = []
    for hole_state, rule in sorted(holes):
        choices = positions[rule].copy()
        rng.shuffle(choices)
        found = 0
        for recipient, step in choices:
            program = train[recipient]
            if program.trace[step] == hole_state or not suffix_is_seen(
                    program, step, hole_state, table, holes):
                continue
            seen_states = [s for s in range(STATES)
                           if s != program.trace[step] and (s, rule) not in holes
                           and donors[(step, s)] and
                           suffix_is_seen(program, step, s, table, holes)]
            if not seen_states or not donors[(step, hole_state)]:
                continue
            seen_state = rng.choice(seen_states)
            result.append(Intervention(
                recipient, rng.choice(donors[(step, hole_state)]),
                rng.choice(donors[(step, seen_state)]), step,
                hole_state, seen_state))
            found += 1
            if found == per_hole:
                break
        if found != per_hole:
            raise RuntimeError(f"only {found}/{per_hole} interventions for {(hole_state, rule)}")
    rng.shuffle(result)
    for item in result:
        recipient = train[item.recipient]
        assert train[item.donor].trace[item.step] == item.hole_state
        assert train[item.seen_donor].trace[item.step] == item.seen_state
        assert (item.hole_state, recipient.rules[item.step]) in holes
        assert (item.seen_state, recipient.rules[item.step]) not in holes
        assert suffix_is_seen(recipient, item.step, item.hole_state, table, holes)
        assert suffix_is_seen(recipient, item.step, item.seen_state, table, holes)
        assert recipient.trace[item.step] not in (item.hole_state, item.seen_state)
    return result


def test_programs(table, holes, seed: int, per_stratum: int) -> dict[str, list[Program]]:
    rng = random.Random(seed)
    groups = {}
    used = set()
    for depth in (4, 5, 6, 8, 10):
        counts = (0, 1, 2, 3, 4, 5) if depth == 10 else (0, 1)
        for hole_count in counts:
            key = f"depth_{depth}_holes_{hole_count}"
            rows = []
            attempts = 0
            while len(rows) < per_stratum:
                attempts += 1
                if attempts > per_stratum * 1000:
                    raise RuntimeError(f"cannot fill {key}")
                initial = rng.randrange(STATES)
                hole_steps = set(rng.sample(range(depth), hole_count))
                state, rules = initial, []
                for step in range(depth):
                    options = [r for r in range(RULES)
                               if ((state, r) in holes) == (step in hole_steps)]
                    if not options:
                        break
                    rule = rng.choice(options)
                    rules.append(rule)
                    state = table[state][rule]
                if len(rules) != depth or (initial, tuple(rules)) in used:
                    continue
                used.add((initial, tuple(rules)))
                rows.append(traced(initial, tuple(rules), table))
            groups[key] = rows
    assert all(sum((s, r) in holes for s, r in zip(p.trace, p.rules)) == int(key.split('_')[-1])
               for key, rows in groups.items() for p in rows)
    return groups


def final_logits(model: HardTransition, initial: torch.Tensor,
                 rules: torch.Tensor, *, return_trace: bool = False):
    current = F.one_hot(initial.long(), STATES).float()
    trace = []
    for step in range(rules.shape[1]):
        raw = model(current, rules[:, step])
        if return_trace:
            trace.append(raw.argmax(-1))
        if step + 1 < rules.shape[1]:
            current = hard_state(raw)
    return (raw, torch.stack(trace, -1)) if return_trace else raw


def cf_loss(model: HardTransition, items: list[Intervention], indices: torch.Tensor,
            train: list[Program], table, *, seen: bool, direct: bool) -> torch.Tensor:
    selected = [items[i] for i in indices.tolist()]
    states = [item.seen_state if seen else item.hole_state for item in selected]
    rules = [train[item.recipient].rules[item.step] for item in selected]
    if direct:
        logits = model(torch.tensor(states), torch.tensor(rules))
        targets = torch.tensor([table[s][r] for s, r in zip(states, rules)])
        return F.cross_entropy(logits, targets)
    grouped = defaultdict(list)
    for index, item in enumerate(selected):
        recipient = train[item.recipient]
        grouped[(len(recipient.rules), item.step)].append(index)
    total = torch.zeros((), dtype=torch.float32)
    for (depth, step), group in grouped.items():
        starts = torch.tensor([states[i] for i in group])
        suffix = torch.tensor([train[selected[i].recipient].rules[step:depth]
                               for i in group])
        target = torch.tensor([traced(states[i], tuple(suffix[j].tolist()), table).trace[-1]
                               for j, i in enumerate(group)])
        total = total + F.cross_entropy(final_logits(model, starts, suffix),
                                       target, reduction="sum")
    return total / len(selected)


@torch.no_grad()
def pair_score(model: HardTransition, table, holes) -> dict:
    states = torch.tensor([s for s in range(STATES) for _ in range(RULES)])
    rules = torch.tensor([r for _ in range(STATES) for r in range(RULES)])
    guesses = model(states, rules).argmax(-1).tolist()
    result = {"seen": [0, 0], "holes": [0, 0], "hole_predictions": {}}
    for state, rule, guess in zip(states.tolist(), rules.tolist(), guesses):
        group = "holes" if (state, rule) in holes else "seen"
        result[group][0] += int(guess == table[state][rule])
        result[group][1] += 1
        if group == "holes":
            result["hole_predictions"][f"{state}:{rule}"] = guess
    for group in ("seen", "holes"):
        correct, n = result[group]
        result[group] = {"correct": correct, "n": n, "accuracy": correct / n}
    return result


@torch.no_grad()
def program_score(model: HardTransition, groups: dict[str, list[Program]]) -> dict:
    result = {}
    for key, rows in groups.items():
        guesses, paths = [], []
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            initial = torch.tensor([p.initial for p in batch])
            rules = torch.tensor([p.rules for p in batch])
            final, trace = final_logits(model, initial, rules, return_trace=True)
            guesses.extend(final.argmax(-1).tolist())
            paths.extend(trace.tolist())
        correct = sum(guess == p.trace[-1] for guess, p in zip(guesses, rows))
        exact = sum(path == list(p.trace[1:]) for path, p in zip(paths, rows))
        result[key] = {"correct": correct, "n": len(rows),
                       "accuracy": correct / len(rows),
                       "exact_trajectory_correct": exact,
                       "exact_trajectory_accuracy": exact / len(rows)}
    return result


def run(seed: int, *, smoke: bool, out: Path) -> dict:
    started = time.monotonic()
    torch.set_num_threads(1)
    table, holes = machine(seed)
    train, val = factual_programs(table, holes, seed + 1)
    per_hole = 4 if smoke else 32
    items = interventions(train, table, holes, seed + 2, per_hole)
    tests = test_programs(table, holes, seed + 3, 24 if smoke else 256)
    pairs = [(s, r) for s in range(STATES) for r in range(RULES)
             if (s, r) not in holes]
    states = torch.tensor([s for s, _ in pairs])
    rules = torch.tensor([r for _, r in pairs])
    targets = torch.tensor([table[s][r] for s, r in pairs])
    # Every factual training example is a step of a disjoint shallow program.
    factual = [(state, rule, table[state][rule]) for p in train
               for state, rule in zip(p.trace, p.rules)]
    factual_counts = Counter(factual)
    fact_states = torch.tensor([s for s, _, _ in factual])
    fact_rules = torch.tensor([r for _, r, _ in factual])
    fact_targets = torch.tensor([next_state for _, _, next_state in factual])
    assert set(factual) == {(s, r, table[s][r]) for s, r in pairs}
    base_seed = 11
    train_seeds = (base_seed,) if smoke else (11, 23, 37, 41, 53)
    results = {}
    warm_steps, main_steps = (1200, 20) if smoke else (2000, 400)
    for train_seed in train_seeds:
        torch.manual_seed(train_seed)
        model = HardTransition()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0)
        warm_done = 0
        for warm_done in range(1, warm_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(states, rules), targets)
            loss.backward()
            optimizer.step()
            if warm_done % 25 == 0:
                with torch.no_grad():
                    accuracy = (model(states, rules).argmax(-1) == targets).float().mean().item()
                    if accuracy == 1.0 and loss.item() < 0.05:
                        break
        if accuracy < 0.99:
            raise RuntimeError(f"seen-pair warm start failed: {accuracy:.3f}")
        warm_model = copy.deepcopy(model.state_dict())
        warm_optimizer = copy.deepcopy(optimizer.state_dict())
        generator = torch.Generator().manual_seed(train_seed + 10000)
        factual_order = torch.randint(len(factual), (main_steps, 128), generator=generator)
        cf_indices = []
        while len(cf_indices) < main_steps * 128:
            cf_indices.extend(torch.randperm(len(items), generator=generator).tolist())
        cf_order = torch.tensor(cf_indices[:main_steps * 128]).reshape(main_steps, 128)
        actual_holes = Counter()
        actual_positions = Counter()
        for index in cf_order.flatten().tolist():
            item = items[index]
            recipient = train[item.recipient]
            actual_holes[f"{item.hole_state}:{recipient.rules[item.step]}"] += 1
            actual_positions[f"{len(recipient.rules)}:{item.step}"] += 1
        val_groups = {f"depth_{depth}": [p for p in val if len(p.rules) == depth]
                      for depth in (1, 2, 3)}
        results[str(train_seed)] = {"warm": {"steps": warm_done,
                                              "seen_pair_accuracy": accuracy,
                                              "validation": program_score(model, val_groups)},
                                    "intervention_exposure": {
                                        "order_sha256": hashlib.sha256(cf_order.numpy().tobytes()).hexdigest(),
                                        "actual_hole_min": min(actual_holes.values()),
                                        "actual_hole_max": max(actual_holes.values()),
                                        "actual_hole_counts": dict(sorted(actual_holes.items())),
                                        "actual_position_counts": dict(sorted(actual_positions.items()))},
                                    "arms": {}}
        for arm in ARMS:
            model = HardTransition()
            model.load_state_dict(warm_model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
            optimizer.load_state_dict(copy.deepcopy(warm_optimizer))
            for group in optimizer.param_groups:
                group["lr"] = 0.001
            for step in range(main_steps):
                batch = factual_order[step]
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(fact_states[batch], fact_rules[batch]),
                                       fact_targets[batch])
                if arm != "State":
                    loss = loss + cf_loss(model, items, cf_order[step], train, table,
                                          seen=arm == "State+IIT-seen",
                                          direct=arm == "State+CF-Aug")
                loss.backward()
                optimizer.step()
            result = {"pairs": pair_score(model, table, holes),
                      "shallow_validation": program_score(model, val_groups),
                      "long_programs": program_score(model, tests)}
            results[str(train_seed)]["arms"][arm] = result
    hole_exposure = defaultdict(int)
    position_exposure = defaultdict(int)
    for item in items:
        p = train[item.recipient]
        hole_exposure[f"{item.hole_state}:{p.rules[item.step]}"] += 1
        position_exposure[f"{len(p.rules)}:{item.step}"] += 1
    assert set(hole_exposure) == {f"{s}:{r}" for s, r in holes}
    assert set(hole_exposure.values()) == {per_hole}
    # Last-step IIT and CF-Aug have literally the same raw logits and targets.
    terminal = next(item for item in items
                    if item.step == len(train[item.recipient].rules) - 1)
    torch.manual_seed(0)
    check_model = HardTransition()
    direct_model = copy.deepcopy(check_model)
    terminal_index = torch.tensor([items.index(terminal)])
    indirect = cf_loss(check_model, items, terminal_index, train, table,
                       seen=False, direct=False)
    direct = cf_loss(direct_model, items, terminal_index, train, table,
                     seen=False, direct=True)
    assert torch.allclose(indirect, direct, atol=1e-7)
    indirect.backward()
    direct.backward()
    assert all(torch.allclose(a.grad, b.grad, atol=1e-7)
               for a, b in zip(check_model.parameters(), direct_model.parameters()))
    digest = hashlib.sha256(json.dumps([item.__dict__ for item in items],
                                       sort_keys=True).encode()).hexdigest()
    report = {
        "design": {"states": STATES, "rules": RULES, "hole_fraction": len(holes) / (STATES * RULES),
                   "data_seed": seed, "train_seeds": train_seeds,
                   "hidden": 128, "warm_lr": 0.005, "main_lr": 0.001,
                   "factual_weight": 1,
                   "counterfactual_weight": 1, "warm_step_cap": warm_steps,
                   "main_steps": main_steps, "batch_size": 128,
                   "factual_loss": "gold-current-state local transition CE",
                   "iit_loss": "gold donor state, recipient suffix final raw-logit CE",
                   "cf_aug_loss": "same intervention tuple, local transition raw-logit CE",
                   "selection": "fixed main steps; warm stop from seen-pair accuracy/loss only"},
        "support": {"table": table, "holes": sorted(holes), "n_seen_pairs": len(pairs),
                    "n_train_programs": len(train), "n_validation_programs": len(val),
                    "train_program_depth_counts": {str(d): sum(len(p.rules) == d for p in train)
                                                   for d in (1, 2, 3)},
                    "factual_step_count": len(factual),
                    "factual_pair_frequency_min": min(factual_counts.values()),
                    "factual_pair_frequency_max": max(factual_counts.values()),
                    "intervention_count": len(items), "intervention_tuple_sha256": digest,
                    "iit_cf_aug_same_tuples": True,
                    "pool_hole_count_min": min(hole_exposure.values()),
                    "pool_hole_count_max": max(hole_exposure.values()),
                    "pool_position_counts": dict(sorted(position_exposure.items()))},
        "results": results, "seconds": time.monotonic() - started,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--out", type=Path, default=Path("runs/support_hole/results.json"))
    args = parser.parse_args()
    result = run(args.seed, smoke=args.smoke, out=args.out)
    print(json.dumps({"out": str(args.out), "seconds": result["seconds"],
                      "pool_hole_count": [result["support"]["pool_hole_count_min"],
                                          result["support"]["pool_hole_count_max"]],
                      "actual_hole_exposure": {
                          seed: [value["intervention_exposure"]["actual_hole_min"],
                                 value["intervention_exposure"]["actual_hole_max"]]
                          for seed, value in result["results"].items()},
                      "seeds": list(result["results"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
