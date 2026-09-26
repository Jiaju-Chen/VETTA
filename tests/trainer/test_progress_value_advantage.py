import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_progress_value_outcome_advantage


def test_progress_value_advantage_uses_uid_step_baseline_and_remaining_steps():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    token_level_rewards = torch.zeros_like(response_mask)

    advantages, returns = compute_progress_value_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["task", "task", "task", "task"], dtype=object),
        traj_index=np.array(["traj-a", "traj-b", "traj-a", "traj-b"], dtype=object),
        step_id=np.array([0, 0, 1, 1], dtype=np.int32),
        episode_lengths=np.array([4, 2, 4, 2], dtype=np.float32),
        episode_rewards=np.array([10, 10, 10, 10], dtype=np.float32),
        reward_scale=1.0,
        length_penalty=0.1,
        remaining_penalty=0.2,
        baseline_mode="uid_step",
        min_group_size=2,
        normalize_by_std=False,
        whiten=False,
    )

    # target = reward - 0.1 * episode_length - 0.2 * remaining_after_action
    # step 0 targets: [9.0, 9.6], mean 9.30 -> centered [-0.30, 0.30]
    # step 1 targets: [9.2, 9.8], mean 9.50 -> centered [-0.30, 0.30]
    expected_adv = torch.tensor([-0.30, 0.30, -0.30, 0.30]).unsqueeze(-1).repeat(1, 3)
    expected_returns = torch.tensor([9.0, 9.6, 9.2, 9.8]).unsqueeze(-1).repeat(1, 3)

    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(returns, expected_returns, atol=1e-6)


def test_progress_value_advantage_falls_back_to_uid_baseline_when_step_group_is_small():
    response_mask = torch.ones((3, 2), dtype=torch.float32)
    token_level_rewards = torch.zeros_like(response_mask)

    advantages, _ = compute_progress_value_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=np.array(["task", "task", "task"], dtype=object),
        traj_index=np.array(["traj-a", "traj-b", "traj-c"], dtype=object),
        step_id=np.array([0, 1, 1], dtype=np.int32),
        episode_lengths=np.array([1, 3, 5], dtype=np.float32),
        episode_rewards=np.array([10, 10, 10], dtype=np.float32),
        reward_scale=1.0,
        length_penalty=1.0,
        remaining_penalty=0.0,
        baseline_mode="uid_step",
        min_group_size=2,
        normalize_by_std=False,
        whiten=False,
    )

    expected = torch.tensor([2.0, 1.0, -1.0]).unsqueeze(-1).repeat(1, 2)
    assert torch.allclose(advantages, expected, atol=1e-6)
