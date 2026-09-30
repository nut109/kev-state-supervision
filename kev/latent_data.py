"""Small, fully symbolic composition tasks for the latent scratchpad experiment.

Each world supplies two questions with the same evidence and candidate set. Worlds,
not questions, are assigned to splits. Nothing here calls a language model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PEOPLE = tuple(f"Person{i:02d}" for i in range(64))
INSTITUTIONS = tuple(f"Institute{i:02d}" for i in range(16))
CITIES = tuple(f"City{i:02d}" for i in range(16))
OBJECTS = tuple(f"Object{i:02d}" for i in range(16))
COLORS = ("amber", "blue", "coral", "green", "indigo", "orange", "purple", "silver",
          "teal", "violet", "white", "yellow")
PREDICATES = ("red", "warm", "active", "ready", "bright", "calm", "brave", "quick",
              "kind", "alert", "steady", "safe")
GATES = ("large", "licensed", "listed", "clear", "local", "open", "trusted", "verified",
         "early", "marked", "senior", "available")

SELF_RELATIONS = ("advisor", "parent", "friend")
TAILS = ((), ("works_at",), ("owns",), ("works_at", "located_in"), ("owns", "color"))
HELD_RELATION_PAIRS = (("advisor", "friend"), ("parent", "advisor"), ("friend", "parent"))
OPERATORS = ("copy", "invert", "and", "or")
HELD_RULE_PAIRS = (("invert", "and"), ("or", "invert"), ("and", "copy"))
DEFAULT_SIZES = {"train": 10_000, "val": 2_000, "id_test": 2_000, "ood_test": 2_000}
CAUSAL_DEPTHS = {"train": (1, 2, 3), "val": (1, 2, 3),
                 "stress_val": (4, 5, 6, 8, 10), "id_test": (1, 2, 3),
                 "ood_test": (4, 5, 6, 8, 10)}


def solve_relations(edges: list[list[str]], start: str, path: list[str]) -> list[str]:
    lookup = {(source, relation): target for source, relation, target in edges}
    steps = []
    current = start
    for relation in path:
        current = lookup[current, relation]
        steps.append(current)
    return steps


def solve_rules(initial: bool, operators: list[str], gates: list[bool | None]) -> list[bool]:
    value = initial
    steps = []
    for op, gate in zip(operators, gates, strict=True):
        if op == "copy":
            pass
        elif op == "invert":
            value = not value
        elif op == "and":
            value = value and gate
        elif op == "or":
            value = value or gate
        else:
            raise ValueError(op)
        steps.append(value)
    return steps


def _path(rng: random.Random, depth: int, ood: bool, rule: bool) -> list[str]:
    if rule:
        choices, held = OPERATORS, HELD_RULE_PAIRS
        prefix, tail = [], []
    else:
        held = HELD_RELATION_PAIRS
        tail = list(rng.choice([t for t in TAILS if len(t) <= depth]))
        prefix = []
        choices = SELF_RELATIONS
        if ood:
            tail = list(rng.choice([t for t in TAILS if len(t) <= depth - 2]))
    n = depth - len(tail)
    if ood:
        prefix = list(rng.choice(held))
    while len(prefix) < n:
        next_item = rng.choice(choices)
        if ood or not prefix or (prefix[-1], next_item) not in held:
            prefix.append(next_item)
    return prefix + tail


def _paired(rows: list[dict], world_id: str) -> list[dict]:
    a, b = rows
    a["pair_id"] = b["pair_id"] = world_id
    a["pair_example_id"], b["pair_example_id"] = b["id"], a["id"]
    a["counterfactual_label"], b["counterfactual_label"] = b["label"], a["label"]
    a["counterfactual_target"] = a["options"][b["label"]]
    b["counterfactual_target"] = b["options"][a["label"]]
    for recipient, donor in ((a, b), (b, a)):
        valid = []
        for i, (own, swapped) in enumerate(zip(recipient["steps"][:-1], donor["steps"][:-1], strict=True)):
            if recipient["family"] == "relation":
                final = solve_relations(recipient["edges"], swapped, recipient["path"][i + 1:])[-1]
                if own != swapped and final == recipient["counterfactual_target"]:
                    valid.append(i)
            else:
                own_truth, donor_truth = own.endswith("=true"), swapped.endswith("=true")
                gates = next(x["gates"] for x in recipient["rule_inputs"]
                             if x["subject"] == recipient["subject"])
                final = solve_rules(donor_truth, recipient["path"][i + 1:], gates[i + 1:])[-1]
                if own_truth != donor_truth and recipient["options"][recipient["counterfactual_label"]] == ("Yes" if final else "No"):
                    valid.append(i)
        recipient["intervention_steps"] = valid
    return rows


def _relation_world(rng: random.Random, world_id: str, depth: int, ood: bool,
                    relation_index: int) -> list[dict]:
    path = _path(rng, depth, ood, rule=False)
    pools = {"person": iter(rng.sample(PEOPLE, len(PEOPLE))),
             "institution": iter(rng.sample(INSTITUTIONS, len(INSTITUTIONS))),
             "city": iter(rng.sample(CITIES, len(CITIES))),
             "object": iter(rng.sample(OBJECTS, len(OBJECTS))),
             "color": iter(rng.sample(COLORS, len(COLORS)))}
    target_type = {"advisor": "person", "parent": "person", "friend": "person",
                   "works_at": "institution", "located_in": "city",
                   "owns": "object", "color": "color"}
    text = {"advisor": "{0}'s advisor is {1}.", "parent": "{0}'s parent is {1}.",
            "friend": "{0}'s friend is {1}.", "works_at": "{0} works at {1}.",
            "located_in": "{0} is in {1}.", "owns": "{0} owns {1}.",
            "color": "{0} is {1}."}
    edges, chains, sentences = [], [], []
    for _ in range(4):
        chain = [next(pools["person"])]
        for relation in path:
            target = next(pools[target_type[relation]])
            edges.append([chain[-1], relation, target])
            sentences.append(text[relation].format(chain[-1], target))
            chain.append(target)
        chains.append(chain)
    rng.shuffle(sentences)
    state = "Facts:\n" + "\n".join(sentences)
    options = [chain[-1] for chain in chains]
    rng.shuffle(options)
    candidate_values = list(dict.fromkeys(value for chain in chains for value in chain[1:]))
    rng.shuffle(candidate_values)
    positions = (relation_index % 4, (relation_index + 1) % 4)
    rows = []
    for suffix, position in zip("ab", positions, strict=True):
        target = options[position]
        chain = next(chain for chain in chains if chain[-1] == target)
        rows.append({"id": f"{world_id}-{suffix}", "world_id": world_id,
                     "family": "relation", "depth": depth, "state": state,
                     "question": f"Starting from {chain[0]}, follow {' then '.join(path)}. Which value is reached?",
                     "options": options.copy(), "label": position, "steps": chain[1:],
                     "candidate_values": candidate_values.copy(),
                     "template": "relation:" + ">".join(path), "path": path.copy(),
                     "start": chain[0], "mechanism_pair": True,
                     "edges": edges})
    return _paired(rows, world_id)


def _rule_inputs(rng: random.Random, operators: list[str], final: bool) -> tuple[bool, list[bool | None]]:
    value = final
    gates: list[bool | None] = [None] * len(operators)
    for i in range(len(operators) - 1, -1, -1):
        op = operators[i]
        if op == "invert":
            value = not value
        elif op == "and":
            value, gates[i] = (True, True) if value else rng.choice(((False, False), (False, True), (True, False)))
        elif op == "or":
            value, gates[i] = (False, False) if not value else rng.choice(((False, True), (True, False), (True, True)))
    return value, gates


def _rule_world(rng: random.Random, world_id: str, depth: int, ood: bool,
                mechanism_pair: bool) -> list[dict]:
    operators = _path(rng, depth, ood, rule=True)
    while not mechanism_pair and not any(op in ("and", "or") for op in operators):
        operators = _path(rng, depth, ood, rule=True)
    predicates = rng.sample(PREDICATES, depth + 1)
    gate_names = rng.sample(GATES, depth)
    subjects = rng.sample(PEOPLE, 4)
    rules = []
    for i, op in enumerate(operators):
        before, after, gate = predicates[i], predicates[i + 1], gate_names[i]
        condition = {"copy": f"is {before}", "invert": f"is not {before}",
                     "and": f"is {before} and is {gate}",
                     "or": f"is {before} or is {gate}"}[op]
        rules.append(f"A subject is {after} exactly when that subject {condition}.")
    if mechanism_pair:
        # Shared neutral gates make every donor state a valid suffix intervention.
        neutral_gates = [True if op == "and" else False if op == "or" else None
                         for op in operators]
        first_initial = bool(sum(op == "invert" for op in operators) % 2)
        profiles = [(initial, neutral_gates.copy(), solve_rules(initial, operators, neutral_gates))
                    for initial in (first_initial, not first_initial)]
    else:
        # Sample varied queried gates, requiring one nonneutral gate to change
        # both an intermediate state and the final answer.
        def causal_gate(profile: tuple[bool, list[bool | None], list[bool]]) -> bool:
            initial, gates, values = profile
            for i, (op, gate) in enumerate(zip(operators, gates, strict=True)):
                before = initial if i == 0 else values[i - 1]
                changed_step = (op == "and" and before and gate is False) or (op == "or" and not before and gate is True)
                if changed_step:
                    flipped = gates.copy()
                    flipped[i] = not gate
                    if solve_rules(initial, operators, flipped)[-1] != values[-1]:
                        return True
            return False

        for _ in range(256):
            profiles = []
            for final in (False, True):
                initial, gates = _rule_inputs(rng, operators, final)
                profiles.append((initial, gates, solve_rules(initial, operators, gates)))
            if any(causal_gate(profile) for profile in profiles):
                break
        else:
            # A constructive fallback avoids a rare rejection-sampling failure.
            pivot = min(i for i, op in enumerate(operators) if op in ("and", "or"))
            gates = [True if op == "and" else False if op == "or" else None for op in operators]
            gates[pivot] = not gates[pivot]
            before = operators[pivot] == "and"
            initial = before ^ bool(sum(op == "invert" for op in operators[:pivot]) % 2)
            values = solve_rules(initial, operators, gates)
            other_initial, other_gates = _rule_inputs(rng, operators, not values[-1])
            other_values = solve_rules(other_initial, operators, other_gates)
            profiles = [None, None]
            profiles[int(values[-1])] = (initial, gates, values)
            profiles[int(other_values[-1])] = (other_initial, other_gates, other_values)
    for final in (False, True):
        initial, gates = _rule_inputs(rng, operators, final)
        profiles.append((initial, gates, solve_rules(initial, operators, gates)))
    values_by_subject = []
    inputs = []
    facts = []
    for subject, (initial, gates, values) in zip(subjects, profiles, strict=True):
        inputs.append({"subject": subject, "initial": initial, "gates": gates})
        values_by_subject.append(values)
        facts.append(f"{subject} is {'not ' if not initial else ''}{predicates[0]}.")
        for op, gate_name, gate_value in zip(operators, gate_names, gates, strict=True):
            if op in ("and", "or"):
                facts.append(f"{subject} is {'not ' if not gate_value else ''}{gate_name}.")
    rng.shuffle(facts)
    rng.shuffle(rules)
    state = "Facts:\n" + "\n".join(facts) + "\nRules:\n" + "\n".join(rules)
    symbolic = [[f"{subject}:{predicate}={'true' if value else 'false'}"
                 for predicate, value in zip(predicates[1:], values, strict=True)]
                for subject, values in zip(subjects, values_by_subject, strict=True)]
    candidate_values = [f"{subject}:{predicate}={truth}"
                        for subject in subjects for predicate in predicates[1:]
                        for truth in ("true", "false")]
    rng.shuffle(candidate_values)
    options = ["No", "Yes"]
    rng.shuffle(options)
    rows = []
    for suffix, j in zip("ab", (0, 1), strict=True):
        rows.append({"id": f"{world_id}-{suffix}", "world_id": world_id,
                     "family": "rule", "depth": depth, "state": state,
                     "question": f"Is {subjects[j]} {predicates[-1]} after applying the rules?",
                     "options": options.copy(),
                     "label": options.index("Yes" if values_by_subject[j][-1] else "No"),
                     "steps": symbolic[j], "candidate_values": candidate_values.copy(),
                     "template": "rule:" + ">".join(operators), "path": operators.copy(),
                     "predicates": predicates.copy(), "subject": subjects[j],
                     "rule_inputs": inputs, "mechanism_pair": mechanism_pair})
    return _paired(rows, world_id)


def generate_splits(seed: int = 0, sizes: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """Build four deterministic, world-disjoint partitions (two rows per world)."""
    sizes = DEFAULT_SIZES if sizes is None else sizes
    rng = random.Random(seed)
    splits = {}
    for split in ("train", "val", "id_test", "ood_test"):
        n = sizes[split]
        if n % 2:
            raise ValueError("Each split size must be even because worlds contain paired questions")
        rows = []
        relation_index = 0
        for world_index in range(n // 2):
            family = "relation" if world_index % 2 == 0 else "rule"
            family_index = world_index // 2
            if split == "train":
                depth = (4 + (family_index // 10) % 3 if family_index % 10 == 9
                         else 1 + family_index % 3)
            elif split == "val":
                depth = 1 + (world_index // 2) % 6
            else:
                depth = (4 if split == "ood_test" else 1) + (world_index // 2) % 3
            world_id = f"{split}-{world_index:05d}"
            if family == "relation":
                rows.extend(_relation_world(rng, world_id, depth, split == "ood_test", relation_index))
                relation_index += 1
            else:
                rows.extend(_rule_world(rng, world_id, depth, split == "ood_test",
                                        mechanism_pair=(world_index // 2) % 5 == 0))
        rng.shuffle(rows)
        splits[split] = rows
    return splits


def causal_intervention(recipient: dict, donor: dict, t: int,
                        factor: str = "dynamic") -> dict:
    """Continue A from step t with only B's current value; t=0 precedes step 1.

    The current value is an entity for relation tasks and a truth accumulator for
    rule tasks. A's future path, gates, world, and options always determine the
    answer. No stored trajectory or label is read by this oracle.
    """
    if (recipient["world_id"] != donor["world_id"] or
            recipient["family"] != donor["family"] or
            recipient["depth"] != donor["depth"] or
            not 0 <= t < recipient["depth"] or
            factor not in ("dynamic", "nuisance")):
        raise ValueError("Intervention requires same-world, same-type rows and a valid step/factor")

    if recipient["family"] == "relation":
        def value_at(row: dict) -> str:
            return row["start"] if t == 0 else solve_relations(
                row["edges"], row["start"], row["path"][:t])[-1]

        own, other = value_at(recipient), value_at(donor)
        answer = lambda value: solve_relations(
            recipient["edges"], value, recipient["path"][t:])[-1]
        donor_answer = solve_relations(donor["edges"], donor["start"], donor["path"])[-1]
    else:
        def profile(row: dict) -> dict:
            return next(x for x in row["rule_inputs"] if x["subject"] == row["subject"])

        def value_at(row: dict) -> bool:
            inputs = profile(row)
            return inputs["initial"] if t == 0 else solve_rules(
                inputs["initial"], row["path"][:t], inputs["gates"][:t])[-1]

        own, other = value_at(recipient), value_at(donor)
        gates = profile(recipient)["gates"]
        answer = lambda value: "Yes" if solve_rules(
            value, recipient["path"][t:], gates[t:])[-1] else "No"
        donor_inputs = profile(donor)
        donor_answer = "Yes" if solve_rules(
            donor_inputs["initial"], donor["path"], donor_inputs["gates"])[-1] else "No"

    target = answer(other if factor == "dynamic" else own)
    baseline = answer(own)
    try:
        label = recipient["options"].index(target)
    except ValueError as exc:
        raise ValueError("Intervened state has no recipient option") from exc
    return {"step": t, "factor": factor,
            "recipient_value": own if factor == "dynamic" else recipient["nuisance"],
            "donor_value": other if factor == "dynamic" else donor["nuisance"],
            "target": target, "label": label, "changed": target != baseline,
            "donor_copy": target == donor_answer}


def _causal_relation_world(rng: random.Random, world_id: str, depth: int) -> list[dict]:
    def path() -> list[str]:
        return [rng.choice(SELF_RELATIONS) for _ in range(depth)]

    path_a = path()
    path_b = path()
    while path_b == path_a:
        path_b = path()
    text = {"advisor": "{0}'s advisor is {1}.", "parent": "{0}'s parent is {1}.",
            "friend": "{0}'s friend is {1}."}
    while True:
        nodes = rng.sample(PEOPLE, 4 * (depth + 1))
        layers = [nodes[4 * i:4 * (i + 1)] for i in range(depth + 1)]
        edges = []
        for i, (a, b) in enumerate(zip(path_a, path_b, strict=True)):
            for relation in dict.fromkeys((a, b)):
                targets = rng.sample(layers[i + 1], 4)
                edges.extend([[source, relation, target]
                              for source, target in zip(layers[i], targets, strict=True)])
        options = rng.sample(layers[-1], 4)
        steps_a = solve_relations(edges, layers[0][0], path_a)
        steps_b = solve_relations(edges, layers[0][1], path_b)
        rows = []
        markers = rng.sample(COLORS, 2)
        sentences = [text[relation].format(source, target)
                     for source, relation, target in edges]
        sentences += [f"{layers[0][j]} has marker {markers[j]}." for j in (0, 1)]
        rng.shuffle(sentences)
        state = "Facts:\n" + "\n".join(sentences)
        candidates = list(dict.fromkeys(target for _, _, target in edges))
        rng.shuffle(candidates)
        for suffix, j, current_path, steps in (("a", 0, path_a, steps_a),
                                                ("b", 1, path_b, steps_b)):
            rows.append({"id": f"{world_id}-{suffix}", "world_id": world_id,
                         "family": "relation", "depth": depth, "state": state,
                         "question": f"Starting from {layers[0][j]}, follow {' then '.join(current_path)}. Which value is reached?",
                         "options": options.copy(), "label": options.index(steps[-1]),
                         "steps": steps, "candidate_values": candidates.copy(),
                         "template": "relation:" + ">".join(current_path),
                         "path": current_path, "start": layers[0][j], "edges": edges,
                         "nuisance": markers[j]})
        a, b = rows
        if (a["label"] != b["label"] and
                all(result["changed"] and not result["donor_copy"] for result in
                    (causal_intervention(a, b, 0), causal_intervention(b, a, 0)))):
            return rows


def _causal_rule_world(rng: random.Random, world_id: str, depth: int,
                       final: bool) -> list[dict]:
    def profile() -> tuple[bool, list[str], list[bool | None], list[bool]]:
        path = [rng.choice(OPERATORS) for _ in range(depth)]
        gates = [bool(rng.randrange(2)) if op in ("and", "or") else None for op in path]
        initial = bool(rng.randrange(2))
        return initial, path, gates, solve_rules(initial, path, gates)

    profiles = (profile(), profile())
    if profiles[0][-1][-1] == profiles[1][-1][-1] != final:
        # Boolean dual flips the answer while preserving the iid operator law.
        def dual(item: tuple) -> tuple:
            initial, path, gates, _ = item
            path = [{"and": "or", "or": "and"}.get(op, op) for op in path]
            gates = [None if gate is None else not gate for gate in gates]
            return not initial, path, gates, solve_rules(not initial, path, gates)

        profiles = tuple(dual(item) for item in profiles)

    paths = [profile[1] for profile in profiles]
    predicates = rng.sample(PREDICATES, depth + 1)
    gate_names = rng.sample(GATES, depth)
    subjects = rng.sample(PEOPLE, 2)
    markers = rng.sample(COLORS, 2)
    options = rng.sample(["No", "Yes"], 2)
    inputs, facts, rules, trajectories = [], [], [], []
    for j, (subject, (initial, path, gates, values)) in enumerate(zip(subjects, profiles, strict=True)):
        trajectories.append(values)
        inputs.append({"subject": subject, "initial": initial, "gates": gates})
        facts.extend([f"{subject} is {'not ' if not initial else ''}{predicates[0]}.",
                      f"{subject} has marker {markers[j]}."])
        for i, (op, gate) in enumerate(zip(path, gates, strict=True)):
            before, after, gate_name = predicates[i], predicates[i + 1], gate_names[i]
            condition = {"copy": f"is {before}", "invert": f"is not {before}",
                         "and": f"is {before} and is {gate_name}",
                         "or": f"is {before} or is {gate_name}"}[op]
            rules.append(f"For {subject}: that subject is {after} exactly when that subject {condition}.")
            if gate is not None:
                facts.append(f"{subject} is {'not ' if not gate else ''}{gate_name}.")
    rng.shuffle(facts)
    state = "Facts:\n" + "\n".join(facts) + "\nRules:\n" + "\n".join(rules)
    candidates = [f"{subject}:{predicate}={truth}"
                  for subject in subjects for predicate in predicates[1:]
                  for truth in ("true", "false")]
    rng.shuffle(candidates)
    rows = []
    for suffix, j, path, values in (("a", 0, paths[0], trajectories[0]),
                                    ("b", 1, paths[1], trajectories[1])):
        rows.append({"id": f"{world_id}-{suffix}", "world_id": world_id,
                     "family": "rule", "depth": depth, "state": state,
                     "question": f"Is {subjects[j]} {predicates[-1]} after applying the rules for {subjects[j]}?",
                     "options": options.copy(),
                     "label": options.index("Yes" if values[-1] else "No"),
                     "steps": [f"{subjects[j]}:{predicate}={'true' if value else 'false'}"
                               for predicate, value in zip(predicates[1:], values, strict=True)],
                     "candidate_values": candidates.copy(),
                     "template": "rule:" + ">".join(path), "path": path,
                     "predicates": predicates.copy(), "subject": subjects[j],
                     "rule_inputs": inputs, "nuisance": markers[j]})
    return rows


def generate_causal_splits(seed: int = 0, sizes: dict[str, int] | None = None) -> dict[str, list[dict]]:
    """Depth extrapolation splits with typed, nontrivial same-world interventions."""
    sizes = DEFAULT_SIZES if sizes is None else sizes
    rng = random.Random(seed)
    splits = {}
    for split in CAUSAL_DEPTHS:
        n = sizes.get(split, min(200, sizes["val"]))
        if n % 2:
            raise ValueError("Each split size must be even because worlds contain paired questions")
        rows = []
        family_indices = {"relation": 0, "rule": 0}
        rule_balance = {depth: 0 for depth in CAUSAL_DEPTHS[split]}
        for world_index in range(n // 2):
            family = "relation" if world_index % 2 == 0 else "rule"
            depth_choices = CAUSAL_DEPTHS[split]
            family_index = family_indices[family]
            depth = depth_choices[family_index % len(depth_choices)]
            family_indices[family] += 1
            world_id = f"causal-{split}-{world_index:05d}"
            pair = (_causal_relation_world(rng, world_id, depth)
                    if family == "relation" else
                    _causal_rule_world(rng, world_id, depth,
                                       bool(rule_balance[depth] % 2)))
            a, b = pair
            if family == "rule" and a["label"] == b["label"]:
                rule_balance[depth] += 1
            a["pair_id"] = b["pair_id"] = world_id
            a["pair_example_id"], b["pair_example_id"] = b["id"], a["id"]
            for recipient, donor in ((a, b), (b, a)):
                recipient["intervention_steps"] = []
                recipient["null_intervention_steps"] = []
                recipient["donor_copy_steps"] = []
                for t in range(depth):
                    result = causal_intervention(recipient, donor, t)
                    if result["changed"] and not result["donor_copy"]:
                        recipient["intervention_steps"].append(t)
                    elif result["changed"]:
                        recipient["donor_copy_steps"].append(t)
                    elif not result["changed"]:
                        recipient["null_intervention_steps"].append(t)
            rows.extend(pair)
        rng.shuffle(rows)
        splits[split] = rows
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate latent scratchpad composition data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--protocol", choices=("legacy", "causal"), default="legacy")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generator = generate_causal_splits if args.protocol == "causal" else generate_splits
    for split, rows in generator(args.seed).items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
