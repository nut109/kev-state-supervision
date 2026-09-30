"""Small CPU checks for the source-only, shared-head Kev program parser."""

from copy import deepcopy
from types import SimpleNamespace

import torch
from torch import nn

from kev.latent_data import generate_causal_splits
from kev.latent_iit_encoder import KevRuleEncoder, OnePassKevRuleEncoder, rule_texts


class TinyTokenizer:
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return 1 if token.endswith("prefix|>") else 2

    def __call__(self, text, add_special_tokens=False):
        self.calls.append(text)
        return SimpleNamespace(input_ids=[ord(char) % 128 + 3 for char in text])


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.config = SimpleNamespace(hidden_size=12)
        self.embedding = nn.Embedding(256, 12)
        self.layers = nn.ModuleList(nn.Linear(12, 12) for _ in range(8))

    def forward(self, input_ids, attention_mask, use_cache=False):
        self.calls.append((input_ids.shape[0], input_ids.shape[1]))
        hidden = self.embedding(input_ids).cumsum(1)
        hidden = hidden / torch.arange(1, hidden.shape[1] + 1,
                                       device=hidden.device)[None, :, None]
        for layer in self.layers:
            hidden = torch.tanh(layer(hidden))
        return SimpleNamespace(last_hidden_state=hidden)


def row(depth=3):
    rules = [
        "For Person02: that subject is warm exactly when that subject is red.",
        "For Person02: that subject is safe exactly when that subject is not warm.",
        "For Person02: that subject is bright exactly when that subject is safe and is large.",
    ][:depth]
    return {
        "state": "Facts:\nPerson01 is not red.\nPerson02 is red.\n"
                 "Person02 has marker yellow.\nPerson02 is large.\n"
                 "Rules:\nFor Person01: that subject is warm exactly when that subject is red.\n"
                 + "\n".join(rules),
        "question": "Is Person02 bright after applying the rules for Person02?",
        "options": ["Yes", "No"],
        # Deliberate contradictions: these fields are supervision, not inputs.
        "path": ["or"] * depth,
        "rule_inputs": [{"subject": "Person02", "initial": False, "gates": [None] * depth}],
        "steps": ["false"] * depth,
        "label": 1,
    }


def test_rule_texts_select_subject_and_keep_source_order():
    initial, steps = rule_texts(row())
    assert len(steps) == 3
    assert "Person01" not in initial + "".join(steps)
    assert "marker yellow" in initial
    assert "is warm exactly" in steps[0]
    assert "is safe exactly" in steps[1]
    assert "is bright exactly" in steps[2]


def test_rule_texts_match_generated_causal_program_for_both_subjects():
    sizes = {split: 24 for split in ("train", "val", "stress_val", "id_test", "ood_test")}
    rows = [row for row in generate_causal_splits(47, sizes)["train"]
            if row["family"] == "rule"]
    one_pass = OnePassKevRuleEncoder(device="cpu", unfreeze_last=0,
                                     tokenizer=TinyTokenizer(), backbone=TinyBackbone())
    assert len(rows) == 12
    for world in {row["world_id"] for row in rows}:
        assert len({row["subject"] for row in rows if row["world_id"] == world}) == 2
    for row in rows:
        initial, steps = rule_texts(row)
        subject = row["subject"]
        source_rules = [line for line in row["state"].split("Rules:\n", 1)[1].splitlines()
                        if line.startswith(f"For {subject}: ")]
        assert len(steps) == row["depth"] == len(source_rules)
        _, readouts, _ = one_pass._one_pass_ids(
            row["question"], tuple(row["options"]), row["state"])
        assert len(readouts) == len(source_rules)
        assert source_rules[0] in initial
        assert [step.split("Rule: ", 1)[1].split("\nOperator and gate:", 1)[0]
                for step in steps] == source_rules
        for i, (before, after, operator) in enumerate(zip(
                row["predicates"][:-1], row["predicates"][1:], row["path"], strict=True)):
            expected = {"copy": f"is {before}.", "invert": f"is not {before}.",
                        "and": f"is {before} and is ", "or": f"is {before} or is "}[operator]
            assert f"that subject is {after} exactly when that subject {expected}" in source_rules[i]


def test_forward_pads_variable_depth_and_ignores_private_targets():
    torch.manual_seed(4)
    model = KevRuleEncoder(device="cpu", unfreeze_last=2,
                           tokenizer=TinyTokenizer(), backbone=TinyBackbone())
    source = [row(1), row(3)]
    initial, operators, gates = model(source)
    assert initial.shape == (2, 2)
    assert operators.shape == (2, 3, 4)
    assert gates.shape == (2, 3, 3)
    assert torch.equal(operators[0, 1:], torch.zeros_like(operators[0, 1:]))
    assert torch.equal(gates[0, 1:], torch.zeros_like(gates[0, 1:]))
    poisoned = deepcopy(source)
    for item in poisoned:
        item.update(path=["invert"] * 10, rule_inputs=[], steps=[], label=0,
                    depth=10, family="relation")
    same = model(poisoned)
    for original, changed in zip((initial, operators, gates), same, strict=True):
        torch.testing.assert_close(original, changed)


def test_only_last_two_blocks_and_shared_heads_train():
    torch.manual_seed(7)
    model = KevRuleEncoder(device="cpu", unfreeze_last=2,
                           tokenizer=TinyTokenizer(), backbone=TinyBackbone())
    assert all(not any(p.requires_grad for p in block.parameters())
               for block in model.backbone.layers[:-2])
    assert all(all(p.requires_grad for p in block.parameters())
               for block in model.backbone.layers[-2:])
    assert not model.backbone.embedding.weight.requires_grad
    assert all(parameter.requires_grad for head in
               (model.initial_head, model.operator_head, model.gate_head)
               for parameter in head.parameters())
    repeated = row(2)
    first = "For Person02: that subject is warm exactly when that subject is red."
    repeated["state"] = repeated["state"].replace(
        "For Person02: that subject is safe exactly when that subject is not warm.", first)
    _, operators, gates = model([repeated])
    torch.testing.assert_close(operators[:, 0], operators[:, 1])
    torch.testing.assert_close(gates[:, 0], gates[:, 1])
    (operators.sum() + gates.sum()).backward()
    assert model.backbone.layers[-1].weight.grad is not None
    assert model.backbone.layers[0].weight.grad is None


def test_bf16_checkpoint_keeps_fp32_trainable_master_weights():
    model = KevRuleEncoder(device="cpu", unfreeze_last=2,
                           tokenizer=TinyTokenizer(),
                           backbone=TinyBackbone().to(torch.bfloat16))
    assert model.backbone.embedding.weight.dtype == torch.bfloat16
    assert model.backbone.layers[0].weight.dtype == torch.bfloat16
    assert model.backbone.layers[-1].weight.dtype == torch.float32
    assert model.operator_head.weight.dtype == torch.float32


def test_one_pass_reads_each_complete_source_once_without_private_targets():
    torch.manual_seed(8)
    tok, backbone = TinyTokenizer(), TinyBackbone()
    model = OnePassKevRuleEncoder(device="cpu", unfreeze_last=2,
                                  tokenizer=tok, backbone=backbone)
    source = row(3)
    initial, operators, gates = model([source])
    assert backbone.calls == [(1, backbone.calls[0][1])]
    assert "".join(tok.calls) == (f"Question: {source['question']}\n"
                                  f"Options: {' | '.join(source['options'])}\n"
                                  + source["state"])
    ids, steps, initial_at = model._one_pass_ids(
        source["question"], tuple(source["options"]), source["state"])
    assert len(steps) == 3 and list(steps) == sorted(steps)
    assert initial_at == len(ids) - 1 and steps[-1] < initial_at
    assert initial.shape == (1, 2)
    assert operators.shape == (1, 3, 4)
    assert gates.shape == (1, 3, 3)
    poisoned = deepcopy(source)
    poisoned.update(path=["or"] * 10, rule_inputs=[], steps=[], label=0,
                    depth=10, family="relation")
    other = model([poisoned])
    for original, changed in zip((initial, operators, gates), other, strict=True):
        torch.testing.assert_close(original, changed)


def test_one_pass_batches_rows_and_trains_shared_heads():
    torch.manual_seed(9)
    backbone = TinyBackbone()
    model = OnePassKevRuleEncoder(device="cpu", unfreeze_last=2,
                                  tokenizer=TinyTokenizer(), backbone=backbone)
    initial, operators, gates = model([row(1), row(3)])
    assert len(backbone.calls) == 1 and backbone.calls[0][0] == 2
    assert operators.shape == (2, 3, 4)
    assert gates.shape == (2, 3, 3)
    assert torch.equal(operators[0, 1:], torch.zeros_like(operators[0, 1:]))
    (initial.sum() + operators.sum() + gates.sum()).backward()
    assert backbone.layers[-1].weight.grad is not None
    assert backbone.layers[0].weight.grad is None
    assert model.operator_head.weight.grad is not None
    assert model.gate_head.weight.grad is not None
