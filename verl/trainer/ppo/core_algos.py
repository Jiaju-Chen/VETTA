# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_turn_ppo_gae(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: np.ndarray,
    step_id: np.ndarray,
    gamma: float,
    lam: float,
    whiten_advantages: bool = False,
):
    """Compute GAE over response actions using pre-response critic values.

    Each row contains one generated response. The first valid response position
    predicts the state before that response; environment observations enter the
    following row's context but are never policy actions.
    """
    if len({token_level_rewards.shape, values.shape, response_mask.shape}) != 1:
        raise ValueError("Turn-PPO expects rewards, values, and response mask to have the same shape")
    batch_size = token_level_rewards.size(0)
    if len(traj_index) != batch_size or len(step_id) != batch_size:
        raise ValueError("Turn-PPO trajectory metadata must match batch size")

    with torch.no_grad():
        trajectories = defaultdict(list)
        for row in range(batch_size):
            trajectories[traj_index[row]].append(row)

        advantages = torch.zeros_like(token_level_rewards)
        returns = torch.zeros_like(token_level_rewards)
        value_mask = torch.zeros_like(response_mask, dtype=token_level_rewards.dtype)
        for rows in trajectories.values():
            rows.sort(key=lambda row: int(step_id[row]))
            next_value = 0.0
            next_gae = 0.0
            for row in reversed(rows):
                positions = torch.nonzero(response_mask[row] > 0, as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                boundary = int(positions[0])
                value = values[row, boundary]
                reward = token_level_rewards[row, positions].sum()
                delta = reward + float(gamma) * next_value - value
                next_gae = delta + float(gamma) * float(lam) * next_gae
                advantages[row, positions] = next_gae
                returns[row, boundary] = next_gae + value
                value_mask[row, boundary] = 1
                next_value = value

        if whiten_advantages:
            turn_mask = value_mask.bool()
            turn_advantages = advantages[turn_mask]
            if turn_advantages.numel() > 1:
                mean = turn_advantages.mean()
                std = turn_advantages.std(unbiased=False).clamp_min(1e-8)
                advantages = ((advantages - mean) / std) * response_mask
    return advantages, returns, value_mask


def compute_sao_skip_observation_gae(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: np.ndarray,
    step_id: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
    whiten_advantages: bool = True,
):
    """Compute token-level GAE while skipping environment observations.

    Each batch row is one model-generated environment action. Rows belonging to
    the same trajectory are ordered by ``step_id`` and connected at the action
    boundary: the last valid token of action ``t`` bootstraps from the first
    valid token of action ``t + 1``. Environment observations are not model
    actions and therefore do not enter the GAE recursion or policy mask.

    This is the credit-assignment part of SAO only. It intentionally does not
    implement SAO's asynchronous rollout, rollout-policy importance ratio, or
    value-update schedule.
    """
    if token_level_rewards.shape != values.shape or values.shape != response_mask.shape:
        raise ValueError(
            "SAO skip-observation GAE expects rewards, values, and response_mask "
            f"to have the same shape; got {token_level_rewards.shape}, "
            f"{values.shape}, {response_mask.shape}"
        )

    with torch.no_grad():
        batch_size, response_length = token_level_rewards.shape
        if len(traj_index) != batch_size or len(step_id) != batch_size:
            raise ValueError(
                "SAO skip-observation GAE expects traj_index and step_id to "
                f"match batch size {batch_size}; got {len(traj_index)} and {len(step_id)}"
            )

        # Keep the original batch layout for the actor/critic workers. Only the
        # recurrence order changes, so all metadata remains aligned.
        trajectories = defaultdict(list)
        for row in range(batch_size):
            trajectories[traj_index[row]].append(row)

        advantages = torch.zeros_like(values)
        gamma_value = float(gamma)
        lam_value = float(lam)

        for rows in trajectories.values():
            rows.sort(key=lambda row: int(step_id[row]))
            next_action_first_value = 0.0
            next_gae = 0.0

            for row in reversed(rows):
                valid_positions = torch.nonzero(response_mask[row] > 0, as_tuple=False).flatten().tolist()
                if not valid_positions:
                    continue

                # Reverse through only generated action tokens. The boundary
                # value from the next action is carried into this action's last
                # token, which is the explicit observation skip.
                for position_index in range(len(valid_positions) - 1, -1, -1):
                    position = valid_positions[position_index]
                    is_action_end = position_index == len(valid_positions) - 1
                    if is_action_end:
                        next_value = next_action_first_value
                    else:
                        next_position = valid_positions[position_index + 1]
                        next_value = values[row, next_position]

                    delta = token_level_rewards[row, position] + gamma_value * next_value - values[row, position]
                    next_gae = delta + gamma_value * lam_value * next_gae
                    advantages[row, position] = next_gae

                next_action_first_value = values[row, valid_positions[0]]

        returns = advantages + values
        if whiten_advantages:
            advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


def compute_dual_critic_hybrid_gae(
    token_level_rewards: torch.Tensor,
    token_values: torch.Tensor,
    turn_values: torch.Tensor,
    response_mask: torch.Tensor,
    traj_index: np.ndarray,
    step_id: np.ndarray,
    token_gamma: float,
    token_lam: float,
    turn_gamma: float,
    turn_lam: float,
    token_residual_scale: float = 1.0,
    composition_mode: str = "residual",
    whiten_advantages: bool = True,
):
    """Combine skip-observation token credit with a separate turn critic.

    The token critic supplies within-turn credit. The turn critic is evaluated
    at the first generated-token position, whose hidden state is aligned with
    the observation boundary immediately before the action.

    ``composition_mode`` isolates the actor-credit ablations while leaving both
    critic targets unchanged: ``residual`` uses centered token credit (Luna),
    ``direct`` uses uncentered token credit, and the two ``*_only`` modes select
    one branch without changing critic training.
    """
    shapes = {
        token_level_rewards.shape,
        token_values.shape,
        turn_values.shape,
        response_mask.shape,
    }
    if len(shapes) != 1:
        raise ValueError(
            "Dual-critic hybrid GAE expects rewards, token values, turn values, "
            "and response mask to have the same shape"
        )

    token_advantages, token_returns = compute_sao_skip_observation_gae(
        token_level_rewards=token_level_rewards,
        values=token_values,
        response_mask=response_mask,
        traj_index=traj_index,
        step_id=step_id,
        gamma=token_gamma,
        lam=token_lam,
        whiten_advantages=False,
    )

    with torch.no_grad():
        batch_size = token_level_rewards.shape[0]
        if len(traj_index) != batch_size or len(step_id) != batch_size:
            raise ValueError(
                "Dual-critic hybrid GAE expects traj_index and step_id to "
                f"match batch size {batch_size}; got {len(traj_index)} and {len(step_id)}"
            )

        trajectories = defaultdict(list)
        valid_positions_by_row = {}
        for row in range(batch_size):
            trajectories[traj_index[row]].append(row)
            valid_positions_by_row[row] = torch.nonzero(
                response_mask[row] > 0, as_tuple=False
            ).flatten()

        turn_advantages = torch.zeros_like(token_level_rewards)
        turn_returns = torch.zeros_like(token_level_rewards)
        turn_value_mask = torch.zeros_like(response_mask, dtype=token_level_rewards.dtype)

        for rows in trajectories.values():
            rows.sort(key=lambda row: int(step_id[row]))
            next_turn_value = 0.0
            next_turn_gae = 0.0

            for row in reversed(rows):
                valid_positions = valid_positions_by_row[row]
                if valid_positions.numel() == 0:
                    continue

                boundary_position = int(valid_positions[0])
                current_turn_value = turn_values[row, boundary_position]
                turn_reward = token_level_rewards[row, valid_positions].sum()
                delta = turn_reward + float(turn_gamma) * next_turn_value - current_turn_value
                next_turn_gae = delta + float(turn_gamma) * float(turn_lam) * next_turn_gae

                turn_advantages[row, valid_positions] = next_turn_gae
                turn_returns[row, boundary_position] = next_turn_gae + current_turn_value
                turn_value_mask[row, boundary_position] = 1
                next_turn_value = current_turn_value

        token_residuals = torch.zeros_like(token_advantages)
        for row in range(batch_size):
            valid_positions = valid_positions_by_row[row]
            if valid_positions.numel() == 0:
                continue
            row_advantages = token_advantages[row, valid_positions]
            token_residuals[row, valid_positions] = row_advantages - row_advantages.mean()

        if composition_mode == "residual":
            hybrid_advantages = turn_advantages + float(token_residual_scale) * token_residuals
        elif composition_mode == "direct":
            hybrid_advantages = turn_advantages + float(token_residual_scale) * token_advantages
        elif composition_mode == "token_only":
            hybrid_advantages = token_advantages
        elif composition_mode == "turn_only":
            hybrid_advantages = turn_advantages
        else:
            raise ValueError(
                "Unsupported hybrid advantage composition_mode "
                f"{composition_mode!r}; expected residual, direct, token_only, or turn_only"
            )
        if whiten_advantages:
            hybrid_advantages = verl_F.masked_whiten(hybrid_advantages, response_mask)

    return (
        hybrid_advantages,
        token_returns,
        turn_returns,
        turn_value_mask,
        turn_advantages,
        token_residuals,
    )


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        compute_mean_std_cross_steps: bool
            If True (more stable), the mean and std are computed across steps within one group. 
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def _to_numpy_1d(values, dtype=None):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    values = np.asarray(values)
    if dtype is not None:
        values = values.astype(dtype)
    return values.reshape(-1)


def compute_progress_value_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    step_id: np.ndarray,
    episode_lengths: np.ndarray,
    episode_rewards: np.ndarray,
    reward_scale: float = 1.0,
    length_penalty: float = 0.02,
    remaining_penalty: float = 0.02,
    baseline_mode: str = "uid_step",
    min_group_size: int = 2,
    normalize_by_std: bool = False,
    whiten: bool = True,
    epsilon: float = 1e-6,
):
    """Compute action-level advantages from a lightweight progress value.

    Each row in the agent batch is one environment action. The estimator uses
    the observed episode outcome and length to assign a scalar progress target:

        target_t = reward_scale * R_episode
                   - length_penalty * L_episode
                   - remaining_penalty * max(L_episode - step_t - 1, 0)

    It then subtracts a same-task baseline, preferably among rollouts from the
    same prompt group and the same environment step, and broadcasts the scalar
    advantage to all generated response tokens for that action.
    """
    with torch.no_grad():
        device = token_level_rewards.device
        dtype = token_level_rewards.dtype
        batch_size = token_level_rewards.shape[0]

        uid = _to_numpy_1d(index, dtype=object)
        traj_uid = _to_numpy_1d(traj_index, dtype=object)
        steps = _to_numpy_1d(step_id, dtype=np.float32)
        lengths = _to_numpy_1d(episode_lengths, dtype=np.float32)
        rewards = _to_numpy_1d(episode_rewards, dtype=np.float32)

        if not (len(uid) == len(traj_uid) == len(steps) == len(lengths) == len(rewards) == batch_size):
            raise ValueError(
                "progress_value expects uid, traj_uid, step_id, episode_lengths, "
                f"and episode_rewards to match batch size {batch_size}; got "
                f"{len(uid)}, {len(traj_uid)}, {len(steps)}, {len(lengths)}, {len(rewards)}"
            )

        remaining = np.maximum(lengths - steps - 1.0, 0.0)
        targets_np = reward_scale * rewards - length_penalty * lengths - remaining_penalty * remaining
        centered_np = targets_np.astype(np.float32).copy()

        if baseline_mode not in {"uid_step", "uid", "none"}:
            raise ValueError(f"Unsupported progress_value baseline_mode: {baseline_mode}")

        if baseline_mode != "none":
            group_to_indices = defaultdict(list)
            for i in range(batch_size):
                if baseline_mode == "uid_step":
                    key = (uid[i], int(steps[i]))
                else:
                    key = uid[i]
                group_to_indices[key].append(i)

            fallback_group_to_indices = defaultdict(list)
            if baseline_mode == "uid_step":
                for i in range(batch_size):
                    fallback_group_to_indices[uid[i]].append(i)

            for i in range(batch_size):
                if baseline_mode == "uid_step":
                    key = (uid[i], int(steps[i]))
                    group_indices = group_to_indices[key]
                    if len(group_indices) < min_group_size:
                        group_indices = fallback_group_to_indices[uid[i]]
                else:
                    group_indices = group_to_indices[uid[i]]

                if len(group_indices) >= min_group_size:
                    group_targets = targets_np[group_indices]
                    centered_np[i] = targets_np[i] - float(np.mean(group_targets))
                    if normalize_by_std and len(group_indices) > 1:
                        centered_np[i] = centered_np[i] / (float(np.std(group_targets)) + epsilon)

        advantages = torch.as_tensor(centered_np, dtype=dtype, device=device).unsqueeze(-1) * response_mask
        returns = torch.as_tensor(targets_np, dtype=dtype, device=device).unsqueeze(-1) * response_mask
        if whiten:
            advantages = verl_F.masked_whiten(advantages, response_mask) * response_mask

    return advantages, returns


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, compute_mean_std_cross_steps: bool = True):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_turn_ppo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """Turn-PPO's response-ratio objective with one clip decision per response."""
    if len({old_log_prob.shape, log_prob.shape, advantages.shape, response_mask.shape}) != 1:
        raise ValueError("Turn-PPO expects token log-probs, advantages, and mask to match")
    if loss_agg_mode != "token-mean":
        raise ValueError("Turn-PPO implements the paper's total-token-normalized turn objective")
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    valid_turns = response_mask.sum(dim=-1) > 0
    turn_log_ratio = ((log_prob - old_log_prob) * response_mask).sum(dim=-1)
    # The full response ratio can overflow for long actions; keep exp finite.
    turn_ratio = torch.exp(turn_log_ratio.clamp(min=-20.0, max=20.0))
    turn_advantages = (advantages * response_mask).sum(dim=-1) / response_mask.sum(dim=-1).clamp_min(1)
    unclipped = -turn_advantages * turn_ratio
    clipped = -turn_advantages * turn_ratio.clamp(1 - cliprange_low, 1 + cliprange_high)
    turn_losses = torch.maximum(unclipped, clipped)
    pg_loss = (turn_losses * valid_turns).sum() / response_mask.sum().clamp_min(1)

    pg_clipfrac = (torch.gt(clipped, unclipped).float() * valid_turns).sum() / valid_turns.sum().clamp_min(1)
    ppo_kl = verl_F.masked_mean(old_log_prob - log_prob, response_mask)
    pg_clipfrac_lower = torch.zeros_like(pg_clipfrac)
    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "seq-mean-token-mean",
):
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - cliprange_low, 1 + cliprange_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean")

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def compute_luna_unified_value_loss(
    token_vpreds: torch.Tensor,
    turn_vpreds: torch.Tensor,
    token_values: torch.Tensor,
    turn_values: torch.Tensor,
    token_returns: torch.Tensor,
    turn_returns: torch.Tensor,
    token_mask: torch.Tensor,
    turn_mask: torch.Tensor,
    cliprange_value: float,
    turn_loss_coef: float = 1.0,
    loss_agg_mode: str = "token-mean",
):
    """Train Luna's two value heads with one shared critic forward pass."""
    token_vf_loss, token_vf_clipfrac = compute_value_loss(
        vpreds=token_vpreds,
        values=token_values,
        returns=token_returns,
        response_mask=token_mask,
        cliprange_value=cliprange_value,
        loss_agg_mode=loss_agg_mode,
    )
    turn_vf_loss, turn_vf_clipfrac = compute_value_loss(
        vpreds=turn_vpreds,
        values=turn_values,
        returns=turn_returns,
        response_mask=turn_mask,
        cliprange_value=cliprange_value,
        loss_agg_mode=loss_agg_mode,
    )
    total_vf_loss = token_vf_loss + float(turn_loss_coef) * turn_vf_loss
    return (
        total_vf_loss,
        token_vf_loss,
        token_vf_clipfrac,
        turn_vf_loss,
        turn_vf_clipfrac,
    )


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
