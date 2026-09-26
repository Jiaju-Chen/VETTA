import json

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.advantage_case_study import dump_advantage_case_studies


class _Tokenizer:
    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{token_id}" for token_id in token_ids]

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        pieces = {
            11: "<think>",
            12: "reasoning</think>",
            21: "<action>",
            22: "click[item]",
            23: "</action>",
        }
        return "".join(pieces.get(token_id, str(token_id)) for token_id in token_ids)


def test_dump_advantage_case_study_preserves_residual_composition(tmp_path):
    data = DataProto.from_single_dict(
        {
            "responses": torch.tensor([[11, 12, 0], [21, 22, 23]]),
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            "values": torch.tensor([[0.2, 0.3, 0.0], [0.1, 0.1, 0.1]]),
            "returns": torch.tensor([[1.2, 2.3, 0.0], [1.1, 2.1, 3.1]]),
            "turn_advantages": torch.tensor([[0.5, 0.5, 0.0], [1.0, 1.0, 1.0]]),
            "token_residuals": torch.tensor([[-0.5, 0.5, 0.0], [-1.0, 0.0, 1.0]]),
            "advantages": torch.tensor([[-1.0, 2.0, 0.0], [-2.0, 1.0, 4.0]]),
            "token_level_rewards": torch.zeros((2, 3)),
            "old_policy_entropies": torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0]]),
            "traj_uid": np.array(["trajectory-a", "trajectory-a"], dtype=object),
            "step_id": np.array([0, 1]),
            "uid": np.array(["task-a", "task-a"], dtype=object),
            "action_text": np.array(
                ["<think>reasoning</think>", "<action>click[item]</action>"],
                dtype=object,
            ),
            "is_action_valid": np.array([True, True]),
            "episode_success": np.array([1.0, 1.0]),
            "episode_rewards": np.array([1.0, 1.0]),
            "episode_lengths": np.array([2, 2]),
        }
    )

    output_path = dump_advantage_case_studies(
        data=data,
        tokenizer=_Tokenizer(),
        output_dir=str(tmp_path),
        global_step=17,
        token_residual_scale=3.0,
        composition_mode="residual",
        whiten_advantages=True,
        max_trajectories=1,
        successful_only=True,
    )

    payload = json.loads(open(output_path, encoding="utf-8").read())
    turns = payload["trajectories"][0]["turns"]
    assert turns[0]["token_advantages_raw"] == [1.0, 2.0]
    assert turns[0]["final_advantage_pre_whitening"] == [-1.0, 2.0]
    assert turns[1]["final_advantage_pre_whitening"] == [-2.0, 1.0, 4.0]
    assert abs(turns[0]["residual_mean"]) < 1e-7
    assert abs(turns[1]["final_mean_minus_turn"]) < 1e-7
    assert turns[0]["token_regions"] == ["reasoning", "closing_or_eos"]
    assert turns[1]["token_regions"] == ["action", "action", "closing_or_eos"]
    assert turns[1]["old_policy_token_entropy"] == [3.0, 4.0, 5.0]
    assert turns[1]["format"]["extracted_action"] == "click[item]"
