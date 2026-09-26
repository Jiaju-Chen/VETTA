import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_dual_critic_hybrid_gae


def compute(rewards, token_values, turn_values, mask, *, token_lam=1.0):
    return compute_dual_critic_hybrid_gae(
        token_level_rewards=rewards,
        token_values=token_values,
        turn_values=turn_values,
        response_mask=mask,
        traj_index=np.asarray(["trajectory-0"] * rewards.shape[0], dtype=object),
        step_id=np.arange(rewards.shape[0], dtype=np.int32),
        token_gamma=1.0,
        token_lam=token_lam,
        turn_gamma=1.0,
        turn_lam=1.0,
        whiten_advantages=False,
    )


def test_terminal_reward_reaches_all_actions_and_turn_boundaries():
    rewards = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    token_values = torch.zeros_like(rewards)
    turn_values = torch.zeros_like(rewards)
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)

    advantages, token_returns, turn_returns, turn_mask, _, residuals = compute(
        rewards, token_values, turn_values, mask
    )

    expected_actions = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    expected_boundaries = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    torch.testing.assert_close(advantages, expected_actions)
    torch.testing.assert_close(token_returns, expected_actions)
    torch.testing.assert_close(turn_returns, expected_boundaries)
    torch.testing.assert_close(turn_mask, expected_boundaries)
    torch.testing.assert_close(residuals, torch.zeros_like(residuals))


def test_hybrid_advantage_preserves_token_residuals_around_turn_mean():
    rewards = torch.tensor([[0.0, 1.0]])
    token_values = torch.tensor([[0.2, 0.7]])
    turn_values = torch.tensor([[0.4, 0.0]])
    mask = torch.ones_like(rewards)

    advantages, _, turn_returns, turn_mask, turn_advantages, residuals = compute(
        rewards, token_values, turn_values, mask, token_lam=0.0
    )

    torch.testing.assert_close(turn_advantages, torch.tensor([[0.6, 0.6]]))
    torch.testing.assert_close(residuals, torch.tensor([[0.1, -0.1]]))
    torch.testing.assert_close(advantages, torch.tensor([[0.7, 0.5]]))
    torch.testing.assert_close(advantages.mean(dim=-1), torch.tensor([0.6]))
    torch.testing.assert_close(turn_returns, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(turn_mask, torch.tensor([[1.0, 0.0]]))


def test_turn_gae_does_not_use_padding_as_a_boundary():
    rewards = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    token_values = torch.zeros_like(rewards)
    turn_values = torch.tensor([[9.0, 0.25, 7.0, 0.0]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 0.0]])

    _, _, turn_returns, turn_mask, turn_advantages, _ = compute(
        rewards, token_values, turn_values, mask
    )

    torch.testing.assert_close(turn_mask, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    torch.testing.assert_close(turn_returns, torch.tensor([[0.0, 1.0, 0.0, 0.0]]))
    torch.testing.assert_close(turn_advantages, torch.tensor([[0.0, 0.75, 0.75, 0.0]]))


def test_turn_gae_accepts_bfloat16_critic_values_with_float_rewards():
    rewards = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    token_values = torch.zeros_like(rewards, dtype=torch.bfloat16)
    turn_values = torch.zeros_like(rewards, dtype=torch.bfloat16)
    mask = torch.ones_like(rewards)

    advantages, _, turn_returns, _, _, _ = compute(
        rewards, token_values, turn_values, mask
    )

    assert advantages.dtype == torch.float32
    assert turn_returns.dtype == torch.float32
    torch.testing.assert_close(advantages, torch.ones_like(rewards))
