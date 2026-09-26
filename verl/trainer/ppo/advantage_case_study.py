"""Export per-token Luna credit signals for reproducible case-study figures."""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _scalar_metadata(data: DataProto, key: str, row: int, default: Any = None) -> Any:
    values = data.non_tensor_batch.get(key)
    if values is None:
        return default
    return _to_jsonable(values[row])


def _rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
_CLOSING_MARKERS = ("</think>", "</action>", "<|im_end|>", "<|endoftext|>")


def _classify_token_regions(token_text: list[str]) -> list[str]:
    """Label generated pieces by overlap with reasoning/action/tag spans."""
    joined = "".join(token_text)
    lowered = joined.lower()
    boundaries = []
    offset = 0
    for piece in token_text:
        boundaries.append((offset, offset + len(piece)))
        offset += len(piece)

    think_spans = [match.span() for match in _THINK_BLOCK.finditer(joined)]
    action_spans = [match.span() for match in _ACTION_BLOCK.finditer(joined)]

    def overlaps(span: tuple[int, int], ranges: list[tuple[int, int]]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in ranges)

    regions = []
    for piece, span in zip(token_text, boundaries):
        lowered_piece = piece.lower()
        if any(marker in lowered_piece for marker in _CLOSING_MARKERS):
            regions.append("closing_or_eos")
        elif overlaps(span, action_spans):
            regions.append("action")
        elif overlaps(span, think_spans):
            regions.append("reasoning")
        elif "<think" in lowered_piece or "<action" in lowered_piece:
            regions.append("control_tag")
        elif "im_end" in lowered_piece or "endoftext" in lowered_piece:
            regions.append("closing_or_eos")
        else:
            regions.append("other")
    return regions


def _format_diagnostics(action_text: str) -> dict[str, Any]:
    think_blocks = list(_THINK_BLOCK.finditer(action_text))
    action_blocks = list(_ACTION_BLOCK.finditer(action_text))
    extracted_action = action_blocks[0].group(1).strip().lower() if action_blocks else None
    return {
        "has_complete_think_block": bool(think_blocks),
        "has_complete_action_block": bool(action_blocks),
        "think_block_count": len(think_blocks),
        "action_block_count": len(action_blocks),
        "extracted_action": extracted_action,
        "reasoning_characters": sum(len(match.group(0)) for match in think_blocks),
        "trailing_characters_after_action": (
            len(action_text) - action_blocks[0].end() if action_blocks else None
        ),
    }


def dump_advantage_case_studies(
    data: DataProto,
    tokenizer,
    output_dir: str,
    global_step: int,
    token_residual_scale: float,
    composition_mode: str,
    whiten_advantages: bool,
    max_trajectories: int = 8,
    successful_only: bool = True,
    source_metadata: dict[str, Any] | None = None,
) -> str:
    """Write successful trajectories and all credit components to one JSON file.

    The exported ``final_advantage_pre_whitening`` is reconstructed from the
    exact actor-credit formula. It is kept separate from ``actor_advantage`` so
    a paper figure cannot accidentally mix pre- and post-whitening quantities.
    """
    required = {
        "responses",
        "prompts",
        "response_mask",
        "values",
        "returns",
        "turn_advantages",
        "token_residuals",
        "advantages",
        "token_level_rewards",
    }
    missing = sorted(required.difference(data.batch.keys()))
    if missing:
        raise KeyError(f"Cannot export advantage case study; missing batch keys: {missing}")

    traj_uids = data.non_tensor_batch.get("traj_uid")
    step_ids = data.non_tensor_batch.get("step_id")
    if traj_uids is None or step_ids is None:
        raise KeyError("Advantage case study requires traj_uid and step_id metadata")

    grouped_rows: dict[str, list[int]] = defaultdict(list)
    for row, traj_uid in enumerate(traj_uids):
        grouped_rows[str(traj_uid)].append(row)

    tensor_keys = set(required)
    if "old_policy_entropies" in data.batch:
        tensor_keys.add("old_policy_entropies")
    tensors = {key: data.batch[key].detach().cpu() for key in tensor_keys}
    candidates: list[tuple[tuple[float, float, float], dict[str, Any]]] = []

    for traj_uid, rows in grouped_rows.items():
        rows.sort(key=lambda row: int(step_ids[row]))
        success = float(_scalar_metadata(data, "episode_success", rows[0], 0.0) or 0.0)
        if successful_only and success <= 0:
            continue

        turns = []
        all_residuals: list[float] = []
        turn_levels: list[float] = []
        invariant_errors: list[float] = []

        for row in rows:
            valid_positions = torch.nonzero(tensors["response_mask"][row] > 0, as_tuple=False).flatten()
            if valid_positions.numel() == 0:
                continue

            positions = valid_positions.tolist()
            token_ids = tensors["responses"][row, valid_positions].tolist()
            token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
            token_text = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]
            token_regions = _classify_token_regions(token_text)

            token_values = tensors["values"][row, valid_positions]
            token_advantages = tensors["returns"][row, valid_positions] - token_values
            token_residuals = tensors["token_residuals"][row, valid_positions]
            turn_advantages = tensors["turn_advantages"][row, valid_positions]

            if composition_mode == "residual":
                final_pre_whitening = turn_advantages + token_residual_scale * token_residuals
            elif composition_mode == "direct":
                final_pre_whitening = turn_advantages + token_residual_scale * token_advantages
            elif composition_mode == "token_only":
                final_pre_whitening = token_advantages
            elif composition_mode == "turn_only":
                final_pre_whitening = turn_advantages
            else:
                raise ValueError(f"Unsupported composition mode {composition_mode!r}")

            residual_list = token_residuals.tolist()
            turn_level = float(turn_advantages[0].item())
            final_list = final_pre_whitening.tolist()
            all_residuals.extend(residual_list)
            turn_levels.append(turn_level)
            invariant_errors.append(abs(sum(final_list) / len(final_list) - turn_level))

            action_text = str(
                _scalar_metadata(
                    data,
                    "action_text",
                    row,
                    tokenizer.decode(token_ids, skip_special_tokens=True),
                )
            )
            turn_record = {
                    "step_id": int(step_ids[row]),
                    "prompt_text": tokenizer.decode(tensors["prompts"][row], skip_special_tokens=True),
                    "action_text": action_text,
                    "is_action_valid": bool(_scalar_metadata(data, "is_action_valid", row, True)),
                    "response_token_count": len(token_ids),
                    "response_hit_max_length": len(token_ids) == tensors["responses"].shape[-1],
                    "response_positions": positions,
                    "token_ids": token_ids,
                    "token_pieces": token_pieces,
                    "token_text": token_text,
                    "token_regions": token_regions,
                    "token_rewards": tensors["token_level_rewards"][row, valid_positions].tolist(),
                    "token_values": token_values.tolist(),
                    "token_advantages_raw": token_advantages.tolist(),
                    "token_residuals_centered": residual_list,
                    "turn_advantage": turn_level,
                    "final_advantage_pre_whitening": final_list,
                    "actor_advantage": tensors["advantages"][row, valid_positions].tolist(),
                    "residual_mean": float(token_residuals.mean().item()),
                    "final_mean_minus_turn": float(final_pre_whitening.mean().item() - turn_level),
                    "format": _format_diagnostics(action_text),
                }
            if "old_policy_entropies" in tensors:
                turn_record["old_policy_token_entropy"] = tensors["old_policy_entropies"][
                    row, valid_positions
                ].tolist()
            turns.append(turn_record)

        if not turns:
            continue

        episode_reward = float(_scalar_metadata(data, "episode_rewards", rows[0], 0.0) or 0.0)
        record = {
            "trajectory_id": traj_uid,
            "uid": str(_scalar_metadata(data, "uid", rows[0], "")),
            "episode_success": success,
            "episode_reward": episode_reward,
            "episode_length": int(_scalar_metadata(data, "episode_lengths", rows[0], len(turns)) or len(turns)),
            "turns": turns,
            "selection_statistics": {
                "token_residual_rms": _rms(all_residuals),
                "turn_advantage_range": max(turn_levels) - min(turn_levels),
                "max_composition_invariant_error": max(invariant_errors),
            },
        }
        rank = (success, episode_reward, record["selection_statistics"]["token_residual_rms"])
        candidates.append((rank, record))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [record for _, record in candidates[:max_trajectories]]
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"step_{global_step:04d}_advantage_cases.json")
    payload = {
        "schema_version": 2,
        "source": source_metadata or {},
        "global_step": global_step,
        "composition": {
            "mode": composition_mode,
            "token_residual_scale": token_residual_scale,
            "whiten_advantages": whiten_advantages,
            "formula": "A_final_pre = A_turn + alpha * (A_token - mean_turn(A_token))",
        },
        "selection": {
            "successful_only": successful_only,
            "max_trajectories": max_trajectories,
            "candidate_trajectories": len(candidates),
            "ranking": "success, episode_reward, token_residual_rms (descending)",
        },
        "trajectories": selected,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Dumped {len(selected)} advantage case-study trajectories to {output_path}")
    return output_path
