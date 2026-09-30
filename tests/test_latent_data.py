from collections import Counter
from copy import deepcopy

from kev.latent_data import (HELD_RELATION_PAIRS, HELD_RULE_PAIRS, PEOPLE,
                             causal_intervention, generate_causal_splits,
                             generate_splits, solve_relations, solve_rules)


SIZES = {split: 96 for split in ("train", "val", "id_test", "ood_test")}


def test_splits_are_reproducible_world_disjoint_and_depth_controlled():
    splits = generate_splits(7, SIZES)
    assert splits == generate_splits(7, SIZES)
    assert {split: len(rows) for split, rows in splits.items()} == SIZES
    worlds = [{r["world_id"] for r in rows} for rows in splits.values()]
    assert sum(map(len, worlds)) == len(set().union(*worlds))
    assert {1, 2, 3} <= {r["depth"] for r in splits["train"]} <= set(range(1, 7))
    assert {r["depth"] for r in splits["val"]} == {1, 2, 3, 4, 5, 6}
    assert {r["depth"] for r in splits["id_test"]} == {1, 2, 3}
    assert {r["depth"] for r in splits["ood_test"]} == {4, 5, 6}
    for split, rows in splits.items():
        for family in ("relation", "rule"):
            depths = {r["depth"] for r in rows if r["family"] == family}
            if split == "train":
                assert {1, 2, 3} <= depths <= set(range(1, 7))
                family_rows = [r for r in rows if r["family"] == family]
                assert abs(sum(r["depth"] >= 4 for r in family_rows) - len(family_rows) / 10) <= 2
            else:
                assert depths == ({1, 2, 3, 4, 5, 6} if split == "val" else
                                  {4, 5, 6} if split == "ood_test" else {1, 2, 3})


def test_full_default_train_is_ninety_ten_per_family():
    splits = generate_splits(0)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 10_000, "val": 2_000, "id_test": 2_000, "ood_test": 2_000}
    for family in ("relation", "rule"):
        rows = [r for r in splits["train"] if r["family"] == family]
        assert len(rows) == 5_000
        assert sum(r["depth"] >= 4 for r in rows) == 500
        assert {r["depth"] for r in rows} == set(range(1, 7))
        held = HELD_RELATION_PAIRS if family == "relation" else HELD_RULE_PAIRS
        assert all(pair not in held for r in rows for pair in zip(r["path"], r["path"][1:]))


def test_symbolic_solver_matches_labels_and_candidates_do_not_reveal_rule_truth():
    splits = generate_splits(13, SIZES)
    for rows in splits.values():
        for row in rows:
            assert len(row["steps"]) == row["depth"]
            assert all(step in row["candidate_values"] for step in row["steps"])
            assert all(step not in row["question"] for step in row["steps"])
            if row["family"] == "relation":
                assert len(row["options"]) == 4
                steps = solve_relations(row["edges"], row["start"], row["path"])
                assert steps == row["steps"]
                assert row["options"][row["label"]] == steps[-1]
                assert row["options"][row["label"]] not in row["question"]
            else:
                assert set(row["options"]) == {"No", "Yes"}
                inputs = next(x for x in row["rule_inputs"] if x["subject"] == row["subject"])
                values = solve_rules(inputs["initial"], row["path"], inputs["gates"])
                assert row["steps"] == [
                    f"{row['subject']}:{predicate}={'true' if value else 'false'}"
                    for predicate, value in zip(row["predicates"][1:], values, strict=True)]
                assert row["options"][row["label"]] == ("Yes" if values[-1] else "No")
                for target in row["steps"]:
                    opposite = target[:-4] + "false" if target.endswith("true") else target[:-5] + "true"
                    assert opposite in row["candidate_values"]


def test_ood_combinations_are_held_out_and_paired_counterfactuals_are_valid():
    splits = generate_splits(19, SIZES)
    for split, rows in splits.items():
        by_id = {row["id"]: row for row in rows}
        rule_labels = Counter((r["depth"], r["options"][r["label"]])
                              for r in rows if r["family"] == "rule")
        for depth in {r["depth"] for r in rows if r["family"] == "rule"}:
            assert rule_labels[depth, "No"] == rule_labels[depth, "Yes"]
        rule_worlds = {r["world_id"]: r for r in rows if r["family"] == "rule"}
        mechanism_count = sum(r["mechanism_pair"] for r in rule_worlds.values())
        assert abs(mechanism_count - len(rule_worlds) / 5) <= 1
        for row in rows:
            held = HELD_RELATION_PAIRS if row["family"] == "relation" else HELD_RULE_PAIRS
            has_held_pair = any(pair in held for pair in zip(row["path"], row["path"][1:]))
            assert has_held_pair == (split == "ood_test")
            partner = by_id[row["pair_example_id"]]
            assert partner["world_id"] == row["world_id"]
            assert partner["state"] == row["state"]
            assert partner["options"] == row["options"]
            assert partner["candidate_values"] == row["candidate_values"]
            assert row["counterfactual_label"] == partner["label"]
            assert row["counterfactual_target"] == row["options"][partner["label"]]
            assert row["label"] != partner["label"]
            assert row["steps"] != partner["steps"]
            if row["family"] == "relation":
                assert row["intervention_steps"] == list(range(row["depth"] - 1))
                for i in row["intervention_steps"]:
                    final = solve_relations(row["edges"], partner["steps"][i], row["path"][i + 1:])[-1]
                    assert final == row["counterfactual_target"]
            else:
                recipient = next(x for x in row["rule_inputs"] if x["subject"] == row["subject"])
                donor = next(x for x in partner["rule_inputs"] if x["subject"] == partner["subject"])
                if row["mechanism_pair"]:
                    assert recipient["gates"] == donor["gates"]
                    assert row["intervention_steps"] == list(range(row["depth"] - 1))
                else:
                    causal_gate_found = False
                    for profile in (recipient, donor):
                        values = solve_rules(profile["initial"], row["path"], profile["gates"])
                        for i, (op, gate) in enumerate(zip(row["path"], profile["gates"], strict=True)):
                            before = profile["initial"] if i == 0 else values[i - 1]
                            active = (op == "and" and before and gate is False) or (
                                op == "or" and not before and gate is True)
                            if active:
                                flipped = profile["gates"].copy()
                                flipped[i] = not gate
                                causal_gate_found |= solve_rules(profile["initial"], row["path"], flipped)[-1] != values[-1]
                    assert causal_gate_found
                for i in row["intervention_steps"]:
                    donor_value = partner["steps"][i].endswith("=true")
                    final = solve_rules(donor_value, row["path"][i + 1:], recipient["gates"][i + 1:])[-1]
                    assert row["options"][row["counterfactual_label"]] == ("Yes" if final else "No")


def test_causal_splits_are_deterministic_world_disjoint_and_depth_extrapolated():
    sizes = {split: 100 for split in ("train", "val", "stress_val", "id_test", "ood_test")}
    splits = generate_causal_splits(23, sizes)
    assert splits == generate_causal_splits(23, sizes)
    assert {split: len(rows) for split, rows in splits.items()} == sizes
    worlds = [{row["world_id"] for row in rows} for rows in splits.values()]
    assert sum(map(len, worlds)) == len(set().union(*worlds))
    for split, rows in splits.items():
        expected = ({4, 5, 6, 8, 10} if split in ("stress_val", "ood_test")
                    else {1, 2, 3})
        for family in ("relation", "rule"):
            family_rows = [row for row in rows if row["family"] == family]
            assert {row["depth"] for row in family_rows} == expected
            assert any(row["intervention_steps"] for row in family_rows)
        for depth in expected:
            truths = Counter(row["options"][row["label"]] for row in rows
                             if row["family"] == "rule" and row["depth"] == depth)
            assert abs(truths["Yes"] - truths["No"]) <= 2
        assert all("Answer:" not in row["state"] and "Steps:" not in row["state"] for row in rows)


def test_causal_intervention_recomputes_only_dynamic_factor_from_primitives():
    sizes = {split: 60 for split in ("train", "val", "stress_val", "id_test", "ood_test")}
    splits = generate_causal_splits(29, sizes)
    effectful = null = 0
    for rows in splits.values():
        by_id = {row["id"]: row for row in rows}
        for row in rows:
            donor = by_id[row["pair_example_id"]]
            assert row["world_id"] == donor["world_id"]
            assert row["state"] == donor["state"]
            assert row["options"] == donor["options"]
            assert row["nuisance"] != donor["nuisance"]
            assert len(row["steps"]) == row["depth"]
            if row["family"] == "relation":
                own_steps = solve_relations(row["edges"], row["start"], row["path"])
                donor_steps = solve_relations(donor["edges"], donor["start"], donor["path"])
                assert own_steps == row["steps"]
                assert row["options"][row["label"]] == own_steps[-1]
                assert row["path"] != donor["path"]
            else:
                own = next(x for x in row["rule_inputs"] if x["subject"] == row["subject"])
                other = next(x for x in donor["rule_inputs"] if x["subject"] == donor["subject"])
                own_steps = solve_rules(own["initial"], row["path"], own["gates"])
                donor_steps = solve_rules(other["initial"], donor["path"], other["gates"])
                assert row["options"][row["label"]] == ("Yes" if own_steps[-1] else "No")
                # Rules for each subject are serialized in path order.
                rule_lines = [line for line in row["state"].split("Rules:\n")[1].splitlines()
                              if line.startswith(f"For {row['subject']}:")]
                assert len(rule_lines) == row["depth"]
                assert all(f"is {predicate} exactly when" in line for predicate, line in
                           zip(row["predicates"][1:], rule_lines, strict=True))
            for t in range(row["depth"]):
                result = causal_intervention(row, donor, t)
                control = causal_intervention(row, donor, t, "nuisance")
                if row["family"] == "relation":
                    swapped = donor["start"] if t == 0 else donor_steps[t - 1]
                    expected = solve_relations(row["edges"], swapped, row["path"][t:])[-1]
                else:
                    swapped = other["initial"] if t == 0 else donor_steps[t - 1]
                    expected = "Yes" if solve_rules(swapped, row["path"][t:],
                                                    own["gates"][t:])[-1] else "No"
                assert result["target"] == expected
                if row["family"] == "relation":
                    assert result["recipient_value"] in PEOPLE
                    assert result["donor_value"] in PEOPLE
                else:
                    assert isinstance(result["recipient_value"], bool)
                    assert isinstance(result["donor_value"], bool)
                assert result["label"] == row["options"].index(expected)
                assert result["changed"] == (expected != row["options"][row["label"]])
                assert control["target"] == row["options"][row["label"]]
                assert not control["changed"]
                assert (t in row["intervention_steps"]) == (
                    result["changed"] and not result["donor_copy"])
                assert (t in row["donor_copy_steps"]) == (
                    result["changed"] and result["donor_copy"])
                assert (t in row["null_intervention_steps"]) == (not result["changed"])
                effectful += result["changed"]
                null += not result["changed"]
            altered_row, altered_donor = deepcopy(row), deepcopy(donor)
            altered_row["label"] = altered_donor["label"] = -1
            altered_row["steps"] = altered_donor["steps"] = ["corrupted"]
            assert causal_intervention(altered_row, altered_donor, 0) == causal_intervention(row, donor, 0)
    assert effectful > 0 and null > 0


def test_causal_paths_share_composition_distribution_and_use_boolean_gates():
    sizes = {"train": 3_000, "val": 100, "stress_val": 100,
             "id_test": 100, "ood_test": 1_200}
    splits = generate_causal_splits(31, sizes)
    for family, held, expected_pairs in (("relation", HELD_RELATION_PAIRS, 9),
                                         ("rule", HELD_RULE_PAIRS, 16)):
        distributions = []
        for split in ("train", "ood_test"):
            pairs = Counter(pair for row in splits[split] if row["family"] == family
                            for pair in zip(row["path"], row["path"][1:]))
            assert len(pairs) == expected_pairs
            assert all(pairs[pair] > 0 for pair in held)
            n = sum(pairs.values())
            distributions.append({pair: count / n for pair, count in pairs.items()})
        distance = sum(abs(distributions[0].get(pair, 0) - distributions[1].get(pair, 0))
                       for pair in distributions[0].keys() | distributions[1].keys()) / 2
        assert distance < 0.1

    for split, rows in splits.items():
        rule_rows = [row for row in rows if row["family"] == "rule"]
        nonneutral = gated = sensitive = 0
        for row in rule_rows:
            inputs = next(x for x in row["rule_inputs"] if x["subject"] == row["subject"])
            base = solve_rules(inputs["initial"], row["path"], inputs["gates"])[-1]
            for i, (op, gate) in enumerate(zip(row["path"], inputs["gates"], strict=True)):
                if op not in ("and", "or"):
                    continue
                gated += 1
                nonneutral += (op == "and" and gate is False) or (op == "or" and gate is True)
                flipped = inputs["gates"].copy()
                flipped[i] = not gate
                sensitive += solve_rules(inputs["initial"], row["path"], flipped)[-1] != base
        assert gated > 0 and 0.35 < nonneutral / gated < 0.65
        assert sensitive > 0
        for depth in {row["depth"] for row in rule_rows}:
            assert any(row["intervention_steps"] for row in rule_rows if row["depth"] == depth)
