"""CPU checks for five-arm bookkeeping, gradients, and locked-test guard."""

from copy import deepcopy
import json
import re

import pytest
import torch
from torch import nn

from kev.latent_data import generate_causal_splits
from kev.latent_iit_encoder import rule_texts
from kev import latent_iit_trials as trials


class TinyEncoder(nn.Module):
    def __init__(self, device="cpu", unfreeze_last=6):
        super().__init__()
        self.backbone = nn.Embedding(64, 8)
        self.initial_head = nn.Linear(8, 2)
        self.operator_head = nn.Linear(8, 4)
        self.gate_head = nn.Linear(8, 3)
        self.to(device)

    def forward(self, rows):
        indices = torch.tensor([int(re.search(r"Person(\d+)", row["question"]).group(1))
                                for row in rows], device=self.backbone.weight.device)
        hidden = self.backbone(indices)
        depth = max(len(rule_texts(row)[1]) for row in rows)
        return (self.initial_head(hidden),
                self.operator_head(hidden).unsqueeze(1).expand(-1, depth, -1),
                self.gate_head(hidden).unsqueeze(1).expand(-1, depth, -1))


@pytest.fixture(scope="module")
def rule_data():
    sizes = {name: 120 if name == "train" else 20
             for name in ("train", "val", "stress_val", "id_test", "ood_test")}
    return [row for row in generate_causal_splits(20260929, sizes)["train"]
            if row["family"] == "rule"]


def test_model_inputs_ignore_all_private_generator_fields(rule_data):
    model = trials.TrialModel("r3_state", encoder=TinyEncoder())
    rows = rule_data[:4]
    original = model(rows)
    poisoned = deepcopy(rows)
    for row in poisoned:
        row.update(path=[], rule_inputs=[], steps=[], label=-1, depth=10,
                   template="poison", nuisance="poison")
    altered = model(poisoned)
    for name in ("answer", "states", "depths"):
        torch.testing.assert_close(original[name], altered[name])
    for one, two in zip(original["parser"], altered["parser"], strict=True):
        torch.testing.assert_close(one, two)


def test_state_loss_uses_pre_hard_logits_and_backpropagates(rule_data):
    model = trials.TrialModel("r3_state", encoder=TinyEncoder())
    rows = rule_data[:4]
    output = model(rows, return_step_logits=True)
    assert output["step_logits"].requires_grad
    loss = trials.state_loss(output, rows)
    loss.backward()
    assert model.machine.transition.net[0].weight.grad.abs().sum() > 0
    assert model.encoder.initial_head.weight.grad.abs().sum() > 0
    assert model.encoder.operator_head.weight.grad.abs().sum() > 0


def test_effectful_noncopy_sampling_and_donor_gradient(rule_data):
    pool = trials.eligible_interventions(rule_data)
    assert pool
    by_id = {row["id"]: row for row in rule_data}
    recipient_id, donor_id, t, label = next(entry for entries in pool.values()
                                             for entry in entries)
    oracle = trials.causal_intervention(by_id[recipient_id], by_id[donor_id], t)
    assert oracle["changed"] and not oracle["donor_copy"]
    assert label == oracle["label"]
    model = trials.TrialModel("r3_iit", encoder=TinyEncoder())
    loss, terms = trials.train_batch(model, [by_id[recipient_id]],
                                    [(recipient_id, donor_id, t, label)], by_id)
    assert set(terms) == {"answer", "state", "iit"}
    loss.backward()
    donor_subject = int(re.search(r"Person(\d+)", by_id[donor_id]["question"]).group(1))
    assert model.encoder.backbone.weight.grad[donor_subject].abs().sum() > 0


def test_common_init_copies_same_transition_into_every_deep_block(
        tmp_path, monkeypatch):
    monkeypatch.setattr(trials, "OnePassKevRuleEncoder", TinyEncoder)
    for name in ("gold_rule_selection", "onepass_rule_selection", "onepass_rule_stress"):
        (tmp_path / f"{name}.json").write_text(json.dumps({"passed": True}))
    trials.initialize(tmp_path, device="cpu")
    r3 = trials.build_arm(tmp_path, "r3", "cpu")
    deep = trials.build_arm(tmp_path, "deep", "cpu")
    for block in deep.machine.transitions:
        for name, value in r3.machine.transition.state_dict().items():
            torch.testing.assert_close(block.state_dict()[name], value)
    for name, value in r3.encoder.state_dict().items():
        torch.testing.assert_close(deep.encoder.state_dict()[name], value)


def test_cf_metrics_have_separate_complete_denominators_and_lock(rule_data, tmp_path):
    model = trials.TrialModel("r3", encoder=TinyEncoder())
    metrics = trials.score_counterfactual(model, rule_data)
    categories = metrics["all"]
    assert sum(entry["n"] for entry in categories.values()) == sum(
        len(rule_texts(row)[1]) for row in rule_data)
    assert "effectful_noncopy" in categories and "null" in categories
    with pytest.raises(ValueError, match="locked test"):
        trials.evaluate(tmp_path, "r3", "ood_test", "cpu", 8)
    with pytest.raises(ValueError, match="all four arms"):
        trials.evaluate(tmp_path, "r3", "ood_test", "cpu", 8,
                        allow_locked_test=True)
