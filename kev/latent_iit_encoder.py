"""Trainable Kev parsers for the typed, hard causal rule-state experiment.

Only question, options, and visible world text enter either parser. Generator
fields such as ``path`` and ``rule_inputs`` are targets, never encoder inputs.
"""

from __future__ import annotations

from functools import lru_cache
import re

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .model import SPECIAL, pad_id, user_tokens


_SUBJECT = re.compile(r"^Is (\S+) \S+ after applying the rules for \1\?$")
KEV_CHECKPOINT = "jaredpalmer/kev-0.8b@9a45d25eb2ab761841196625383fa1dff0e56c1e"


def rule_texts(row: dict) -> tuple[str, list[str]]:
    """Make initial and ordered-step texts using only fields visible to Kev."""
    question = row["question"]
    match = _SUBJECT.fullmatch(question)
    if match is None:
        raise ValueError(f"not a causal rule question: {question!r}")
    subject = match.group(1)
    facts_section, separator, rules_section = row["state"].partition("\nRules:\n")
    if not separator:
        raise ValueError("state has no Rules section")
    facts = [line for line in facts_section.splitlines()
             if line.startswith(f"{subject} ")]
    rules = [line for line in rules_section.splitlines()
             if line.startswith(f"For {subject}: ")]
    if not facts or not rules:
        raise ValueError(f"missing facts or rules for {subject}")
    fact_text = "\n".join(facts)
    context = (f"Question: {question}\n"
               f"Options: {' | '.join(row['options'])}\n"
               f"Facts:\n{fact_text}\n")
    initial = f"{context}First rule: {rules[0]}\nInitial truth:"
    steps = [f"{context}Rule: {rule}\nOperator and gate:" for rule in rules]
    return initial, steps


def _layers(backbone: nn.Module) -> nn.ModuleList:
    """Locate the text transformer blocks in either supported Qwen wrapper."""
    for path in ("language_model.model.layers", "language_model.layers",
                 "model.layers", "layers"):
        value = backbone
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                break
        if isinstance(value, (nn.ModuleList, list)) and value:
            return value
    raise ValueError("Kev backbone transformer layers were not found")


class KevRuleEncoder(nn.Module):
    """Parse a rule program with merged Kev, tuning its last transformer blocks.

    ``tokenizer`` and ``backbone`` permit a tiny CPU backend in unit tests.
    Production construction loads ``jaredpalmer/kev-0.8b`` with merged LoRA.
    Operator IDs use ``latent_data.OPERATORS`` order, and gate IDs are
    ``(None, False, True)``.  Step logits are zero-padded to the batch's longest
    source-text rule list; callers use source-derived depths to ignore padding.
    """

    def __init__(self, device: str = "cuda", unfreeze_last: int = 6, *,
                 tokenizer=None, backbone: nn.Module | None = None):
        super().__init__()
        if (tokenizer is None) != (backbone is None):
            raise ValueError("provide both tokenizer and backbone, or neither")
        if backbone is None:
            from .checkpoint import Checkpoint, LoadOptions
            tokenizer, kev = Checkpoint(KEV_CHECKPOINT).load(
                device, LoadOptions(dtype=torch.bfloat16, merge=True, backend="torch"))
            backbone = kev.lm  # The original pointer head is intentionally discarded.
        self.tokenizer = tokenizer
        self.backbone = backbone
        layers = _layers(self.backbone)
        if not 0 <= unfreeze_last <= len(layers):
            raise ValueError(f"unfreeze_last must be in [0, {len(layers)}]")
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if unfreeze_last:
            for layer in layers[-unfreeze_last:]:
                # Keep optimizer master weights in fp32; AdamW updates to bf16
                # model weights can round away at the small backbone LR.
                layer.float()
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        width = self.backbone.config.hidden_size
        self.initial_head = nn.Linear(width, 2)
        self.operator_head = nn.Linear(width, 4)
        self.gate_head = nn.Linear(width, 3)
        self.to(device)

    @lru_cache(maxsize=100_000)
    def _token_ids(self, text: str) -> tuple[int, ...]:
        tok = self.tokenizer
        return (tok.convert_tokens_to_ids(SPECIAL[0]),
                *user_tokens(tok, text),
                tok.convert_tokens_to_ids(SPECIAL[4]))

    def _hidden(self, sequences: list[tuple[int, ...]]) -> torch.Tensor:
        """Run one padded backbone batch and return its token representations."""
        device = next(self.backbone.parameters()).device
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), pad_id(self.tokenizer),
                         dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[index, :len(sequence)] = 1
        # Autocast bridges frozen bf16 blocks and trainable fp32 blocks while
        # retaining fp32 master parameters and gradients for the optimizer.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            return self.backbone(input_ids=ids, attention_mask=mask,
                                 use_cache=False).last_hidden_state

    def forward(self, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``initial[B,2]``, ``operators[B,D,4]``, ``gates[B,D,3]``."""
        texts, initial_at, step_at = [], [], []
        for row in rows:
            initial, steps = rule_texts(row)
            initial_at.append(len(texts))
            texts.append(initial)
            step_at.append(list(range(len(texts), len(texts) + len(steps))))
            texts.extend(steps)
        sequences = [self._token_ids(text) for text in texts]
        hidden = self._hidden(sequences)
        device = hidden.device
        last = [len(sequence) - 1 for sequence in sequences]
        h = hidden[torch.arange(len(texts), device=device),
                   torch.tensor(last, device=device)].float()
        initial_logits = self.initial_head(h[initial_at])
        step_indices = [index for positions in step_at for index in positions]
        step_hidden = h[step_indices]
        operator_logits = self.operator_head(step_hidden)
        gate_logits = self.gate_head(step_hidden)
        sizes = [len(positions) for positions in step_at]
        return (initial_logits,
                pad_sequence(operator_logits.split(sizes), batch_first=True),
                pad_sequence(gate_logits.split(sizes), batch_first=True))


class OnePassKevRuleEncoder(KevRuleEncoder):
    """Read the complete visible input once and extract typed rule positions.

    Each row becomes one causal sequence. Question and options precede the
    complete state, so each rule-line readout can attend to its query and facts.
    The final initial-truth readout sees the entire input. Rule positions come
    from visible subject labels and line order; no operation or gate is parsed
    in Python. The shared heads learn those semantics from representations.
    """

    @lru_cache(maxsize=100_000)
    def _one_pass_ids(self, question: str, options: tuple[str, ...], state: str
                      ) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        match = _SUBJECT.fullmatch(question)
        if match is None:
            raise ValueError(f"not a causal rule question: {question!r}")
        subject = match.group(1)
        tok = self.tokenizer
        ids = [tok.convert_tokens_to_ids(SPECIAL[0])]
        ids.extend(user_tokens(tok, f"Question: {question}\nOptions: {' | '.join(options)}\n"))
        line_end = tok.convert_tokens_to_ids(SPECIAL[3])
        steps = []
        in_rules = False
        for line in state.splitlines(keepends=True):
            clean = line.rstrip("\r\n")
            ids.extend(user_tokens(tok, line))
            if clean == "Rules:":
                in_rules = True
            elif in_rules and clean.startswith("For "):
                ids.append(line_end)
                if clean.startswith(f"For {subject}: "):
                    steps.append(len(ids) - 1)
        if not steps:
            raise ValueError(f"no visible rules for {subject}")
        ids.append(tok.convert_tokens_to_ids(SPECIAL[4]))
        return tuple(ids), tuple(steps), len(ids) - 1

    def forward(self, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        packed = [self._one_pass_ids(row["question"], tuple(row["options"]),
                                     row["state"]) for row in rows]
        hidden = self._hidden([item[0] for item in packed])
        device = hidden.device
        initial_at = torch.tensor([item[2] for item in packed], device=device)
        initial_h = hidden[torch.arange(len(rows), device=device), initial_at].float()
        positions = [(i, position) for i, item in enumerate(packed)
                     for position in item[1]]
        row_at, step_at = zip(*positions, strict=True)
        step_h = hidden[torch.tensor(row_at, device=device),
                        torch.tensor(step_at, device=device)].float()
        sizes = [len(item[1]) for item in packed]
        return (self.initial_head(initial_h),
                pad_sequence(self.operator_head(step_h).split(sizes), batch_first=True),
                pad_sequence(self.gate_head(step_h).split(sizes), batch_first=True))
