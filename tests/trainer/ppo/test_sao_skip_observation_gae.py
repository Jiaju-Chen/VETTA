import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_sao_skip_observation_gae


def test_skip_observation_connects_actions_in_one_trajectory():
    rewards = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.float32)

    advantages, returns = compute_sao_skip_observation_gae(
        rewards,
        values,
        mask,
        traj_index=np.asarray(["trajectory-0", "trajectory-0"], dtype=object),
        step_id=np.asarray([0, 1], dtype=np.int32),
        gamma=1.0,
        lam=1.0,
    )

    # The terminal reward at action 1 must propagate through the action
    # boundary to both tokens in action 0. Padding remains irrelevant.
    torch.testing.assert_close(returns, torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]))
    assert torch.isfinite(advantages).all()


def test_skip_observation_does_not_cross_trajectory_boundaries():
    rewards = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ]
    )
    values = torch.zeros_like(rewards)
    mask = torch.ones_like(rewards)

    _, returns = compute_sao_skip_observation_gae(
        rewards,
        values,
        mask,
        traj_index=np.asarray(["trajectory-0", "trajectory-0", "trajectory-1"], dtype=object),
        step_id=np.asarray([0, 1, 0], dtype=np.int32),
        gamma=1.0,
        lam=1.0,
    )

    torch.testing.assert_close(returns[:2], torch.ones((2, 2)))
    torch.testing.assert_close(returns[2], torch.zeros(2))
