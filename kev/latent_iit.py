"""A typed rule-state machine for causal intervention experiments.

The encoder is deliberately outside this module. Its only permitted outputs are
categorical initial truth, ordered operators/gates, depth, and option values.
After the typed input is read, the transition sees one instruction and one
two-value dynamic state at a time. The answer scorer sees only the final state
and recipient options.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def hard_one_hot(logits: torch.Tensor) -> torch.Tensor:
    """Categorical forward values with a softmax straight-through gradient."""
    probabilities = logits.softmax(dim=-1)
    chosen = F.one_hot(probabilities.argmax(dim=-1), logits.shape[-1]).to(probabilities.dtype)
    return chosen + (probabilities - probabilities.detach())


class _RuleStep(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        # Current truth (2), operator (4), gate (none/false/true: 3).
        self.net = nn.Sequential(nn.Linear(9, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, current: torch.Tensor, operator: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((current, operator, gate), dim=-1))


class HardRuleMachine(nn.Module):
    """Shared recurrence or distinct absolute-step transitions over rule state.

    `start_step=t, override_current=donor_state_t` swaps only the dynamic truth
    factor. Operators, gates, depth, and answer options remain the recipient's.
    The step counter is the integer `t`; it is not a trainable latent channel.

    `feedback` changes only the two-value state carried between steps. The
    continuous control carries raw logits, the soft control carries their
    probabilities, and the default hard path retains the original straight-
    through one-hot state. All three use the same transition and supervision
    logits, and the final scorer reads a hard terminal truth in every mode.
    Soft feedback is not normalized a second time at the answer readout.
    The continuous control is deliberately a two-dimensional state, not an
    unconstrained wider hidden vector.

    Operator IDs follow ``latent_data.OPERATORS`` (copy, invert, and, or).
    Gate IDs are 0=None, 1=False, 2=True. Option truth values are 0/1.
    """

    def __init__(self, mode: str = "recurrent", hidden: int = 64,
                 max_depth: int = 10, feedback: str = "hard"):
        super().__init__()
        if mode not in {"recurrent", "deep"}:
            raise ValueError(f"unknown mode: {mode}")
        if feedback not in {"continuous", "soft", "hard"}:
            raise ValueError(f"unknown feedback: {feedback}")
        self.mode = mode
        self.feedback = feedback
        self.max_depth = max_depth
        if mode == "recurrent":
            self.transition = _RuleStep(hidden)
        else:
            self.transitions = nn.ModuleList(_RuleStep(hidden) for _ in range(max_depth))

    @staticmethod
    def gold_logits(indices: torch.Tensor, classes: int, margin: float = 8.0) -> torch.Tensor:
        """Construct typed oracle input logits from categorical indices."""
        return margin * F.one_hot(indices.long(), classes).float()

    @staticmethod
    def score(current: torch.Tensor, option_truth: torch.Tensor) -> torch.Tensor:
        """Match terminal truth to options; the program is not an argument."""
        options = F.one_hot(option_truth.long(), 2).to(current.dtype)
        return 8.0 * (current.unsqueeze(1) * options).sum(dim=-1)

    def rollout(self, initial_logits: torch.Tensor, operator_logits: torch.Tensor,
                gate_logits: torch.Tensor, depths: torch.Tensor,
                option_truth: torch.Tensor, *, loops: int,
                start_step: int = 0, override_current: torch.Tensor | None = None,
                return_step_logits: bool = False):
        """Return answer logits and carried states from the starting state onward.

        `states[:, 0]` is the initial or transplanted state. Once a row reaches
        its depth, subsequent iterations are exact no-ops. A donor may supply
        `override_current`, but never its program or options.
        """
        if loops < 0 or start_step < 0:
            raise ValueError("loops and start_step must be nonnegative")
        if start_step and override_current is None:
            raise ValueError("a nonzero start_step requires override_current")
        if operator_logits.shape[1] > self.max_depth:
            raise ValueError("program exceeds max_depth")
        current = (hard_one_hot(initial_logits) if override_current is None else
                   hard_one_hot(override_current) if self.feedback == "hard" else
                   override_current)
        last_raw = initial_logits if override_current is None else override_current
        states = [current]
        step_logits = []
        program_length = operator_logits.shape[1]
        for step in range(start_step, start_step + loops):
            if step < program_length:
                operator = hard_one_hot(operator_logits[:, step])
                gate = hard_one_hot(gate_logits[:, step])
                block = self.transition if self.mode == "recurrent" else self.transitions[step]
                raw = block(current, operator, gate)
                proposed = (raw if self.feedback == "continuous" else
                            raw.softmax(dim=-1) if self.feedback == "soft" else
                            hard_one_hot(raw))
                active = (depths > step).unsqueeze(-1)
                current = torch.where(active, proposed, current)
                last_raw = torch.where(active, raw, last_raw)
                if return_step_logits:
                    step_logits.append(raw)
            elif return_step_logits:
                step_logits.append(current.new_zeros(current.shape))
            states.append(current)
        answer_state = current if self.feedback == "hard" else hard_one_hot(last_raw)
        result = self.score(answer_state, option_truth), torch.stack(states, dim=1)
        return (*result, torch.stack(step_logits, dim=1)) if return_step_logits else result
