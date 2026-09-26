import numpy as np
import pytest
import torch

from verl.trainer.ppo.core_algos import compute_policy_loss_turn_ppo, compute_turn_ppo_gae, compute_value_loss


def test_turn_gae_uses_next_pre_response_value_and_one_value_target_per_turn():
    rewards = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    values = torch.tensor([[0.2, 9.0, 0.0], [0.4, 8.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])

    advantages, returns, value_mask = compute_turn_ppo_gae(
        rewards, values, mask, np.array(["episode", "episode"]), np.array([0, 1]),
        gamma=0.9, lam=1.0,
    )

    torch.testing.assert_close(advantages, torch.tensor([[0.7, 0.7, 0.0], [0.6, 0.6, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[0.9, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    torch.testing.assert_close(value_mask, torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_turn_gae_does_not_cross_trajectory_boundary_or_padding():
    rewards = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    advantages, returns, value_mask = compute_turn_ppo_gae(
        rewards, values, mask, np.array(["a", "b"]), np.array([0, 0]),
        gamma=1.0, lam=1.0,
    )
    torch.testing.assert_close(advantages, torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]]))
    torch.testing.assert_close(returns, torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    torch.testing.assert_close(value_mask, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def test_turn_ratio_is_product_and_clips_the_whole_response():
    old = torch.zeros((2, 3))
    current = torch.tensor([[0.2, 0.2, 5.0], [0.1, 0.0, 0.0]], requires_grad=True)
    advantages = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    loss, clipfrac, _, _ = compute_policy_loss_turn_ppo(
        old, current, advantages, mask, cliprange=0.2,
    )
    torch.testing.assert_close(loss, -(torch.tensor(1.2) + torch.exp(torch.tensor(0.1))) / 3)
    torch.testing.assert_close(clipfrac, torch.tensor(0.5))
    loss.backward()
    torch.testing.assert_close(current.grad[0], torch.zeros(3))
    torch.testing.assert_close(current.grad[1], torch.tensor([-torch.exp(torch.tensor(0.1)) / 3, 0.0, 0.0]))


def test_turn_critic_loss_only_uses_pre_response_boundary():
    vpreds = torch.tensor([[0.3, 7.0], [0.4, 8.0]])
    old_values = torch.zeros_like(vpreds)
    returns = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    value_mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss, _ = compute_value_loss(vpreds, returns, old_values, value_mask, cliprange_value=10.0)
    torch.testing.assert_close(loss, torch.tensor(((0.3 - 1) ** 2 + 0.4 ** 2) / 2))


def test_turn_policy_loss_rejects_token_aggregation_mode():
    zeros = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="total-token-normalized"):
        compute_policy_loss_turn_ppo(zeros, zeros, zeros, torch.ones_like(zeros), cliprange=0.2, loss_agg_mode="seq-mean-token-mean")
