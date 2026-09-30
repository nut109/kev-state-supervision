"""The IIT path swaps only hard dynamic truth, then follows recipient rules."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from kev.latent_iit import HardRuleMachine


def _gold(initial, operators, gates, options=(0, 1)):
    machine = HardRuleMachine
    return (machine.gold_logits(torch.tensor([initial]), 2),
            machine.gold_logits(torch.tensor([operators]), 4),
            machine.gold_logits(torch.tensor([gates]), 3),
            torch.tensor([len(operators)]), torch.tensor([options]))


class _SymbolicRuleStep(nn.Module):
    """Test-only exact truth table in the learned transition's input format."""

    def forward(self, current, operator, gate):
        truth = current[:, 1]
        gate_true = gate[:, 2]
        outcomes = torch.stack((truth, 1 - truth, truth * gate_true,
                                1 - (1 - truth) * (1 - gate_true)), dim=-1)
        after = (operator * outcomes).sum(dim=-1)
        return 20 * torch.stack((1 - after, after), dim=-1)


def test_transplant_uses_recipient_suffix_and_options():
    machine = HardRuleMachine(max_depth=3)
    machine.transition = _SymbolicRuleStep()
    # Recipient: false --copy--> false --invert--> true.
    recipient = _gold(0, [0, 1], [0, 0], options=(1, 0))
    # Donor: false --invert--> true --copy--> true.
    donor = _gold(0, [1, 0], [0, 0])
    factual, _ = machine.rollout(*recipient, loops=2)
    _, donor_states = machine.rollout(*donor, loops=1)
    counterfactual, states = machine.rollout(
        *recipient, loops=1, start_step=1, override_current=donor_states[:, 1])
    assert factual.argmax(dim=-1).item() == 0  # Recipient's "Yes" option.
    assert counterfactual.argmax(dim=-1).item() == 1  # Recipient suffix inverts donor truth.
    assert states[:, 0].argmax(dim=-1).item() == 1
    assert states[:, 1].argmax(dim=-1).item() == 0


@pytest.mark.parametrize("mode", ["recurrent", "deep"])
def test_hard_states_noop_after_depth_and_program_padding(mode):
    torch.manual_seed(4)
    machine = HardRuleMachine(mode=mode, max_depth=5)
    initial = torch.randn(2, 2)
    operators = torch.randn(2, 5, 4)
    gates = torch.randn(2, 5, 3)
    depths = torch.tensor([2, 4])
    options = torch.tensor([[0, 1], [1, 0]])
    logits, states = machine.rollout(initial, operators, gates, depths, options, loops=8)
    assert logits.shape == (2, 2)
    assert states.shape == (2, 9, 2)
    assert torch.all((states == 0) | (states == 1))
    assert torch.all(states.sum(dim=-1) == 1)
    torch.testing.assert_close(states[0, 2:], states[0, 2].expand_as(states[0, 2:]))
    torch.testing.assert_close(states[1, 4:], states[1, 4].expand_as(states[1, 4:]))
    changed = operators.clone()
    changed[0, 2:] += 100
    changed[1, 4:] -= 100
    other_logits, other_states = machine.rollout(initial, changed, gates, depths, options, loops=8)
    torch.testing.assert_close(other_logits, logits)
    torch.testing.assert_close(other_states, states)


def test_answer_scorer_has_no_program_path():
    machine = HardRuleMachine(max_depth=2)
    initial, operators, gates, depths, options = _gold(0, [1, 1], [0, 0])
    current_false = HardRuleMachine.gold_logits(torch.tensor([0]), 2)
    current_true = HardRuleMachine.gold_logits(torch.tensor([1]), 2)
    false_logits, _ = machine.rollout(initial, operators, gates, depths, options,
                                     loops=0, start_step=2, override_current=current_false)
    true_logits, _ = machine.rollout(initial, -operators, -gates, depths, options,
                                    loops=0, start_step=2, override_current=current_true)
    assert false_logits.argmax(dim=-1).item() == 0
    assert true_logits.argmax(dim=-1).item() == 1
    torch.testing.assert_close(false_logits, machine.score(F.one_hot(torch.tensor([0]), 2).float(), options))
    torch.testing.assert_close(true_logits, machine.score(F.one_hot(torch.tensor([1]), 2).float(), options))


@pytest.mark.parametrize("mode", ["recurrent", "deep"])
def test_straight_through_state_and_program_receive_gradients(mode):
    torch.manual_seed(7)
    machine = HardRuleMachine(mode=mode, max_depth=3)
    initial = torch.randn(3, 2, requires_grad=True)
    operators = torch.randn(3, 3, 4, requires_grad=True)
    gates = torch.randn(3, 3, 3, requires_grad=True)
    logits, states = machine.rollout(initial, operators, gates,
                                    torch.tensor([3, 3, 3]),
                                    torch.tensor([[0, 1]] * 3), loops=3)
    F.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    assert torch.all((states.detach() == 0) | (states.detach() == 1))
    assert initial.grad.abs().sum() > 0
    assert operators.grad.abs().sum() > 0
    assert gates.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in machine.parameters() if p.grad is not None) > 0


def test_deep_has_distinct_absolute_step_blocks():
    machine = HardRuleMachine(mode="deep", max_depth=3)
    assert machine.transitions[0] is not machine.transitions[1]
    assert (machine.transitions[0].net[0].weight.data_ptr() !=
            machine.transitions[1].net[0].weight.data_ptr())
    with pytest.raises(ValueError, match="requires override_current"):
        machine.rollout(*_gold(0, [0], [0]), loops=1, start_step=1)


def test_optional_step_logits_supervise_before_hardening():
    machine = HardRuleMachine(max_depth=3)
    initial = torch.randn(2, 2, requires_grad=True)
    operators = torch.randn(2, 3, 4, requires_grad=True)
    gates = torch.randn(2, 3, 3, requires_grad=True)
    answer, states, raw = machine.rollout(
        initial, operators, gates, torch.tensor([1, 3]),
        torch.tensor([[0, 1], [1, 0]]), loops=3, return_step_logits=True)
    assert answer.shape == (2, 2)
    assert states.shape == (2, 4, 2)
    assert raw.shape == (2, 3, 2)
    assert raw.requires_grad
    F.cross_entropy(raw[1], torch.tensor([1, 0, 1])).backward()
    assert operators.grad is not None and operators.grad.abs().sum() > 0
