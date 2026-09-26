import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from types import SimpleNamespace

from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import compute_luna_unified_value_loss
from verl.workers.critic.dp_critic import DataParallelPPOCritic


class TwoHeadDummyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, **_kwargs):
        positions = torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.float32)
        logits = torch.stack((positions, positions + 10.0), dim=-1)
        logits = logits.unsqueeze(0).expand(input_ids.size(0), -1, -1) * self.scale
        return SimpleNamespace(logits=logits)


def test_unified_value_loss_uses_token_and_turn_masks_independently():
    token_vpreds = torch.zeros((1, 3), requires_grad=True)
    turn_vpreds = torch.zeros((1, 3), requires_grad=True)
    old_values = torch.zeros((1, 3))
    token_returns = torch.tensor([[1.0, 2.0, 999.0]])
    turn_returns = torch.tensor([[3.0, 999.0, 999.0]])
    token_mask = torch.tensor([[1.0, 1.0, 0.0]])
    turn_mask = torch.tensor([[1.0, 0.0, 0.0]])

    total, token_loss, token_clipfrac, turn_loss, turn_clipfrac = (
        compute_luna_unified_value_loss(
            token_vpreds=token_vpreds,
            turn_vpreds=turn_vpreds,
            token_values=old_values,
            turn_values=old_values,
            token_returns=token_returns,
            turn_returns=turn_returns,
            token_mask=token_mask,
            turn_mask=turn_mask,
            cliprange_value=1000.0,
            turn_loss_coef=0.25,
        )
    )

    torch.testing.assert_close(token_loss, torch.tensor(2.5))
    torch.testing.assert_close(turn_loss, torch.tensor(9.0))
    torch.testing.assert_close(total, torch.tensor(4.75))
    torch.testing.assert_close(token_clipfrac, torch.tensor(0.0))
    torch.testing.assert_close(turn_clipfrac, torch.tensor(0.0))

    total.backward()
    torch.testing.assert_close(token_vpreds.grad, torch.tensor([[-1.0, -2.0, 0.0]]))
    torch.testing.assert_close(turn_vpreds.grad, torch.tensor([[-1.5, 0.0, 0.0]]))


def test_two_head_critic_returns_both_values_from_one_forward():
    module = TwoHeadDummyCritic()
    config = OmegaConf.create(
        {
            "model": {"use_remove_padding": False, "num_value_heads": 2},
            "unified_luna_turn_loss_coef": 1.0,
            "ulysses_sequence_parallel_size": 1,
        }
    )
    critic = DataParallelPPOCritic(
        config=config,
        critic_module=module,
        critic_optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
    )
    critic.device_name = "cpu"

    values = critic._forward_micro_batch(
        {
            "input_ids": torch.ones((1, 5), dtype=torch.long),
            "responses": torch.ones((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "position_ids": torch.arange(5).unsqueeze(0),
        }
    )

    assert values.shape == (1, 2, 2)
    torch.testing.assert_close(
        values.float(),
        torch.tensor([[[2.0, 12.0], [3.0, 13.0]]]),
    )


def test_single_head_unified_loss_reuses_one_prediction_for_token_and_turn():
    prediction = torch.zeros((1, 3), requires_grad=True)
    old_values = torch.zeros((1, 3))
    total, token_loss, _, turn_loss, _ = compute_luna_unified_value_loss(
        token_vpreds=prediction,
        turn_vpreds=prediction,
        token_values=old_values,
        turn_values=old_values,
        token_returns=torch.tensor([[1.0, 2.0, 0.0]]),
        turn_returns=torch.tensor([[3.0, 0.0, 0.0]]),
        token_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        turn_mask=torch.tensor([[1.0, 0.0, 0.0]]),
        cliprange_value=1000.0,
        turn_loss_coef=1.0,
    )
    torch.testing.assert_close(token_loss, torch.tensor(2.5))
    torch.testing.assert_close(turn_loss, torch.tensor(9.0))
    torch.testing.assert_close(total, torch.tensor(11.5))
    total.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[-7.0, -2.0, 0.0]]))


@pytest.mark.parametrize(
    ("composition_mode", "expected"),
    [
        ("residual", [[1.0, 3.0, 0.0], [3.0, 7.0, 0.0]]),
        ("direct", [[3.0, 5.0, 0.0], [7.0, 11.0, 0.0]]),
        ("token_only", [[1.0, 3.0, 0.0], [2.0, 6.0, 0.0]]),
        ("turn_only", [[2.0, 2.0, 0.0], [5.0, 5.0, 0.0]]),
    ],
)
def test_luna_actor_advantage_composition_modes(monkeypatch, composition_mode, expected):
    token_advantages = torch.tensor(
        [[1.0, 3.0, 0.0], [2.0, 6.0, 0.0]],
    )

    def fake_token_gae(**_kwargs):
        return token_advantages.clone(), torch.zeros_like(token_advantages)

    monkeypatch.setattr(core_algos, "compute_sao_skip_observation_gae", fake_token_gae)
    rewards = torch.tensor([[2.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    response_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    advantages, *_ = core_algos.compute_dual_critic_hybrid_gae(
        token_level_rewards=rewards,
        token_values=torch.zeros_like(rewards),
        turn_values=torch.zeros_like(rewards),
        response_mask=response_mask,
        traj_index=np.array(["trajectory", "trajectory"]),
        step_id=np.array([0, 1]),
        token_gamma=1.0,
        token_lam=1.0,
        turn_gamma=0.0,
        turn_lam=0.0,
        token_residual_scale=1.0,
        composition_mode=composition_mode,
        whiten_advantages=False,
    )

    torch.testing.assert_close(advantages, torch.tensor(expected))


def test_luna_actor_advantage_rejects_unknown_composition(monkeypatch):
    token_advantages = torch.tensor([[1.0, 3.0]])
    monkeypatch.setattr(
        core_algos,
        "compute_sao_skip_observation_gae",
        lambda **_kwargs: (token_advantages, torch.zeros_like(token_advantages)),
    )

    with pytest.raises(ValueError, match="Unsupported hybrid advantage composition_mode"):
        core_algos.compute_dual_critic_hybrid_gae(
            token_level_rewards=torch.tensor([[1.0, 0.0]]),
            token_values=torch.zeros((1, 2)),
            turn_values=torch.zeros((1, 2)),
            response_mask=torch.ones((1, 2)),
            traj_index=np.array(["trajectory"]),
            step_id=np.array([0]),
            token_gamma=1.0,
            token_lam=1.0,
            turn_gamma=1.0,
            turn_lam=0.95,
            composition_mode="unknown",
            whiten_advantages=False,
        )
